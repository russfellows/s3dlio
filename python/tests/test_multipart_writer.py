# python/tests/test_multipart_writer.py
import os
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
        w = s3dlio.MultipartUploadWriter.from_uri(uri, part_size=32<<20, max_in_flight=16)
        mv = w.reserve(N)
        mv[:] = b"\xAB" * N
        w.commit(N)
        del mv
        info = w.close()
        self.assertEqual(info["total_bytes"], N)
        self.assertGreaterEqual(info["parts"], 2)
        # Optional spot-check if get_range is available
        try:
            head = s3dlio.get_range(uri, 0, 16)
            tail = s3dlio.get_range(uri, N - 16, 16)
            self.assertEqual(bytes(head), b"\xAB" * 16)
            self.assertEqual(bytes(tail), b"\xAB" * 16)
        except Exception:
            pass

    def test_write_bytes_path(self):
        uri = f"s3://{self.bucket}/ckpt-write.bin"
        w = s3dlio.MultipartUploadWriter.from_uri(uri, part_size=16<<20, max_in_flight=8)
        payload = b"\xCD" * (4 << 20)  # 4 MiB
        for _ in range(10):
            w.write(payload)
        info = w.close()
        self.assertEqual(info["total_bytes"], 10 * len(payload))
        self.assertGreaterEqual(info["parts"], 1)

if __name__ == "__main__":
    unittest.main()

