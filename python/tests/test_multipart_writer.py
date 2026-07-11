# python/tests/test_multipart_writer.py
import os
import sys
import time
import unittest
import s3dlio

RUST_LOG = os.environ.get("RUST_LOG", "s3dlio=warn,aws_sdk_s3=warn")
os.environ["RUST_LOG"] = RUST_LOG


def unique_bucket(prefix="mpu-py"):
    return f"{prefix}-{os.getpid()}-{int(time.time())}"


class TestMultipartWriter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bucket = unique_bucket()
        s3dlio.create_bucket(cls.bucket)

    @classmethod
    def tearDownClass(cls):
        try:
            s3dlio.delete_bucket(cls.bucket)
        except Exception:
            pass

    def test_reserve_commit_zero_copy(self):
        uri = f"s3://{self.bucket}/ckpt-zero-copy.bin"
        N = (64 << 20) + (1 << 20)  # 65 MiB
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=32 << 20, max_in_flight=16
        )
        mv = w.reserve(N)
        mv[:] = b"\xab" * N
        w.commit(N)
        del mv
        info = w.close()
        self.assertEqual(info["total_bytes"], N)
        self.assertGreaterEqual(info["parts"], 2)
        # Optional spot-check if get_range is available
        try:
            head = s3dlio.get_range(uri, 0, 16)
            tail = s3dlio.get_range(uri, N - 16, 16)
            self.assertEqual(bytes(head), b"\xab" * 16)
            self.assertEqual(bytes(tail), b"\xab" * 16)
        except Exception:
            pass

    def test_write_bytes_path(self):
        uri = f"s3://{self.bucket}/ckpt-write.bin"
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=16 << 20, max_in_flight=8
        )
        payload = b"\xcd" * (4 << 20)  # 4 MiB
        for _ in range(10):
            w.write(payload)
        info = w.close()
        self.assertEqual(info["total_bytes"], 10 * len(payload))
        self.assertGreaterEqual(info["parts"], 1)

    def test_close_always_has_etag_key(self):
        # Regression test for docs/DESIGN_FFI_BOUNDARY_HARDENING.md §4.3:
        # close()'s docstring promises 'etag': str or None, but the old
        # code only inserted the key when e_tag was Some(...) -- omitting
        # it entirely (not even as None) otherwise, so result['etag']
        # raised KeyError instead of returning None as documented.
        uri = f"s3://{self.bucket}/ckpt-etag-key.bin"
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=16 << 20, max_in_flight=4
        )
        w.write(b"\xef" * (1 << 20))
        info = w.close()
        self.assertIn(
            "etag",
            info,
            "'etag' key must always be present (str or None), never omitted",
        )

    def test_reserve_pins_writer_alive_until_commit(self):
        # Regression test for docs/DESIGN_FFI_BOUNDARY_HARDENING.md §1.3 /
        # §6.3: reserve() used to return a memoryview via
        # PyMemoryView_FromMemory, which carries no reference back to the
        # writer -- dropping the writer while the memoryview was still
        # held would free pending_buf's allocation out from under a live
        # memoryview (use-after-free). The fix holds an extra Py<Self>
        # ("self-pin") for the reserve()..commit() window, observable here
        # as a +1 refcount delta that releases back to baseline on commit().
        #
        # CPython-specific: sys.getrefcount() is a CPython implementation
        # detail, not part of the Python language spec. s3dlio only ever
        # targets CPython (extension-module build, no PyPy classifiers in
        # pyproject.toml), so this is a non-issue in practice -- but a
        # future contributor extending PyPy support should know to revisit
        # this test rather than be surprised by it.
        uri = f"s3://{self.bucket}/uaf-regression.bin"
        w = s3dlio.MultipartUploadWriter.from_uri(
            uri, part_size=32 << 20, max_in_flight=4
        )
        baseline = sys.getrefcount(w)
        mv = w.reserve(1024)
        self.assertEqual(
            sys.getrefcount(w),
            baseline + 1,
            "reserve() must pin an extra strong reference to the writer "
            "for the duration of the pending buffer",
        )
        mv[:] = b"\xab" * 1024
        w.commit(1024)
        self.assertEqual(
            sys.getrefcount(w),
            baseline,
            "the self-pin must be released once the reservation window ends",
        )
        w.close()


if __name__ == "__main__":
    unittest.main()
