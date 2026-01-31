import unittest
from pathlib import Path

import numpy as np

from scripts.python.gaussian_splatting.renderer import load_ply, render_points, write_ppm


class TestGaussianSplatting(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(__file__).resolve().parent / "data" / "3dgs"
        self.output_dir = Path(__file__).resolve().parent / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_render_from_ply(self):
        ply_path = self.data_dir / "sample.ply"
        points = load_ply(ply_path)
        image = render_points(points, width=128, height=128)
        self.assertEqual(image.shape, (128, 128, 3))
        self.assertTrue(np.any(image > 0.1))

        output_path = self.output_dir / "sample_render.ppm"
        write_ppm(output_path, image)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
