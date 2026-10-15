"""``crc32c_pure`` against known answers and a TFRecord stream s3dlio writes (issue #258).

``REFERENCE`` is the byte-exact output of ``build_tfrecord(2, 4, bytes(range(8)))`` (it is
asserted equal in ``tests/test_tfrecord_crc32c_258.rs``), and was independently accepted by
TensorBoard's TFRecord reader, which verifies the masked CRC-32Cs.
"""

import struct
import zlib

from crc32c_pure import crc32c

REFERENCE = bytes.fromhex(
    "040000000000000042455204000102033e9dcad7"
    "04000000000000004245520404050607" "96569695"
)


def unmask(masked):
    rot = (masked - 0xA282EAD8) & 0xFFFFFFFF
    return ((rot >> 17) | (rot << 15)) & 0xFFFFFFFF


def test_known_answers():
    assert crc32c(b"123456789") == 0xE3069283  # not zlib's 0xCBF43926
    assert zlib.crc32(b"123456789") == 0xCBF43926
    assert crc32c(b"") == 0
    assert crc32c(bytes(32)) == 0x8A9136AA  # RFC 3720 B.4
    assert crc32c(b"\xff" * 32) == 0x62A8AB43
    assert crc32c(b"56789", crc32c(b"1234")) == 0xE3069283  # continuation


def test_every_crc_of_the_s3dlio_tfrecord_is_a_masked_crc32c():
    pos, records = 0, []
    while pos < len(REFERENCE):
        (length,) = struct.unpack_from("<Q", REFERENCE, pos)
        (len_crc,) = struct.unpack_from("<I", REFERENCE, pos + 8)
        assert unmask(len_crc) == crc32c(REFERENCE[pos : pos + 8])
        data = REFERENCE[pos + 12 : pos + 12 + length]
        (data_crc,) = struct.unpack_from("<I", REFERENCE, pos + 12 + length)
        assert unmask(data_crc) == crc32c(data)
        # The CRC-32 the verification script used before would have rejected it.
        assert unmask(data_crc) != zlib.crc32(data)
        records.append(data)
        pos += 16 + length
    assert records == [bytes([0, 1, 2, 3]), bytes([4, 5, 6, 7])]
