import unittest
import zfp
import numpy as np


class ZfpTestCase(unittest.TestCase):
    def _test_compress_decompress(self, shape, mode, **kwargs):
        original = np.arange(10000).reshape(shape).astype(np.float32)
        compressed = zfp.compress(original, mode, **kwargs)
        decompressed = np.empty(shape, dtype=np.float32)
        zfp.decompress(compressed, decompressed, mode, **kwargs)
        max_diff = np.abs(original - decompressed).max()
        # print(shape, mode, kwargs, len(compressed) / original.nbytes)
        self.assertLessEqual(max_diff, 0.00001)

    def test_1d_fixed_accuracy(self):
        self._test_compress_decompress((10000,), zfp.CompressionMode.FixedAccuracy, tolerance=0.00001)

    def test_2d_fixed_accuracy(self):
        self._test_compress_decompress((100, 100), zfp.CompressionMode.FixedAccuracy, tolerance=0.00001)

    def test_3d_fixed_accuracy(self):
        self._test_compress_decompress((100, 10, 10), zfp.CompressionMode.FixedAccuracy, tolerance=0.00001)

    def test_4d_fixed_accuracy(self):
        self._test_compress_decompress((10, 10, 10, 10), zfp.CompressionMode.FixedAccuracy, tolerance=0.00001)

    def test_1d_fixed_precision(self):
        self._test_compress_decompress((10000,), zfp.CompressionMode.FixedPrecision, precision=20)

    def test_1d_fixed_rate(self):
        original = np.arange(10000).astype(np.float32)
        compressed = zfp.compress(original, zfp.CompressionMode.FixedRate, rate=4)
        decompressed = np.empty((10000,), dtype=np.float32)
        zfp.decompress(compressed, decompressed, zfp.CompressionMode.FixedRate, rate=4)
        self.assertEqual(len(compressed), 5000)

    def test_2d_fixed_rate(self):
        original = np.arange(10000).reshape((100, 100)).astype(np.float32)
        compressed = zfp.compress(original, zfp.CompressionMode.FixedRate, rate=4)
        decompressed = np.empty((100, 100), dtype=np.float32)
        zfp.decompress(compressed, decompressed, zfp.CompressionMode.FixedRate, rate=4)
        self.assertEqual(len(compressed), 1256)
