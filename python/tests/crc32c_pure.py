"""A small pure-Python CRC-32C (Castagnoli), for the verification scripts.

TFRecord files use CRC-32C (masked) for the length and the data of every record, not
the CRC-32 that ``zlib.crc32`` computes (s3dlio issue #258). The scripts that check
s3dlio's output must use the right polynomial, or they reject correct files.
"""

_POLY = 0x82F63B78  # CRC-32C (Castagnoli), reflected


def _make_table():
    table = []
    for n in range(256):
        crc = n
        for _ in range(8):
            crc = (crc >> 1) ^ _POLY if crc & 1 else crc >> 1
        table.append(crc)
    return tuple(table)


_TABLE = _make_table()


def crc32c(data, crc=0):
    """CRC-32C of ``data`` (``crc`` continues a previous value, as in ``zlib.crc32``)."""
    crc ^= 0xFFFFFFFF
    for byte in data:
        crc = _TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF
