import numpy as np

from mmted.data import CalibratedStandardizer, window_array


def test_window_and_calibration():
    x = np.ones((2, 8, 2), dtype=np.float32); windows = window_array(x, 4)
    assert windows.shape == (4, 4, 2)
    scaled = CalibratedStandardizer(2).fit_transform(x)
    assert np.allclose(scaled, 0)
