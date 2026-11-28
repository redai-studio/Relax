import random
import struct
import unittest

from relax.distributed.weight_sync.codec import MAX_CHUNK_BYTES, _xor, decode, encode


class CodecWindowTests(unittest.TestCase):
    def test_xor_matches_retained_integer_wire_format_at_byte_boundaries(self):
        generator = random.Random(0)
        for size in (0, 1, 7, 8, 65535, 65536, MAX_CHUNK_BYTES - 1, MAX_CHUNK_BYTES):
            with self.subTest(size=size):
                left, right = generator.randbytes(size), generator.randbytes(size)
                expected = (int.from_bytes(left, "little") ^ int.from_bytes(right, "little")).to_bytes(size, "little")
                actual = _xor(left, right)
                self.assertIsInstance(actual, bytes)
                self.assertEqual(actual, expected)
                self.assertEqual(_xor(actual, right), left)

    def test_sparse_wire_format_matches_element_replacement_across_windows(self):
        random_source = random.Random(0)
        for width in (1, 2, 4, 8):
            base = random_source.randbytes(140000 * width)
            current = bytearray(base)
            # Include both sides of each vectorized comparison window.
            indices = [0, 65535, 65536, 65537, 131071, 131072, 139999]
            for index in indices:
                current[index * width] ^= 0x80
            expected = b"".join(
                struct.pack("<I", index) + current[index * width : (index + 1) * width] for index in indices
            )
            kind, payload = encode(bytes(current), base, width)
            self.assertEqual((kind, payload), ("sparse", expected))
            self.assertEqual(decode(kind, payload, base, width, len(base)), current)
            kind, payload = encode(bytes(current), base, width, compression=True)
            self.assertEqual(decode(kind, payload, base, width, len(base)), current)

    def test_dense_replacement_chooses_raw(self):
        for width in (1, 2, 4, 8):
            base = bytes(140000 * width)
            current = b"\xff" * len(base)
            self.assertEqual(encode(current, base, width), ("raw", current))

    def test_float_bit_patterns_keep_signed_zero_and_nan_payloads(self):
        for width, infinity, nan in (
            (2, 0x7F80, 0x7FC1),
            (4, 0x7F800000, 0x7FC00001),
            (8, 0x7FF0000000000000, 0x7FF8000000000001),
        ):
            with self.subTest(width=width):
                sign = 1 << (8 * width - 1)
                patterns = (0, sign, infinity, infinity | sign, nan, nan + 1, nan | sign)
                base = b"".join(value.to_bytes(width, "little") for value in patterns) * 64
                current = bytearray(base)
                replacements = {0: sign, 1: 0, 4: nan + 1, 6: nan}
                for index, value in replacements.items():
                    current[index * width : (index + 1) * width] = value.to_bytes(width, "little")
                expected = b"".join(
                    struct.pack("<I", index) + value.to_bytes(width, "little") for index, value in replacements.items()
                )
                self.assertEqual(encode(bytes(current), base, width), ("sparse", expected))
                self.assertEqual(decode("sparse", expected, base, width, len(base)), current)


if __name__ == "__main__":
    unittest.main()
