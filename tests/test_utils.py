import sys
import os
sys.path.append(os.pardir)

import numpy as np
import unittest

import coropy.utils as cu


class TestUtils(unittest.TestCase):
    """Class for `utils.py` tests."""

    def test_normalize(self):
        self.assertIsNone(
            np.testing.assert_array_equal(cu.normalize([10, 50]), [0., 1.]))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                cu.normalize(range(100)), np.linspace(0, 1, 100), decimal=2))
        self.assertIsNone(
            np.testing.assert_array_equal(cu.normalize([-1, 1]), [0., 1.]))
        self.assertRaises(ValueError, cu.normalize, 5.)
        self.assertRaises(AssertionError, cu.normalize, [[1, 2, 3]])
        self.assertRaises(AssertionError, cu.normalize, [5])
        self.assertRaises(ValueError, cu.normalize, [-np.nan, 50])
        self.assertWarns(RuntimeWarning, cu.normalize, [0, 0, 0])

    def test_restore(self):
        self.assertIsNone(
            np.testing.assert_array_equal(
                cu.restore([0., 1.], [10, 50]), [10., 50.]))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                cu.restore(np.linspace(0, 1, 100), range(100)),
                range(100), 
                decimal=2))
        self.assertRaises(ValueError, cu.restore, [5.], 0)
        self.assertRaises(AssertionError, cu.restore, [[1, 2, 3]], [0, 3])
        self.assertRaises(ValueError, cu.restore, [0, 1], [-np.nan, 50])
    
    def test_moving_average(self):
        a = np.array([1, 2, 3, 4, 5])
        a_2 = np.array([1.5, 2.5, 3.5, 4.5])
        a_5 = np.array([3.])
        self.assertIsNone(
            np.testing.assert_array_equal(cu.moving_average(a, 1), a))
        self.assertIsNone(
            np.testing.assert_array_equal(cu.moving_average(a, 2), a_2))
        self.assertIsNone(
            np.testing.assert_array_equal(cu.moving_average(a, 5), a_5))
        self.assertRaises(ValueError, cu.moving_average, a, 1.5)
        self.assertRaises(ValueError, cu.moving_average, 1)
        self.assertRaises(AssertionError, cu.moving_average, a.reshape(-1, 1))
        self.assertRaises(AssertionError, cu.moving_average, a, 6)

    def test_train_test_split(self):
        a_list = [range(10)]
        a = np.array(a_list)
        a_train, a_test = a[:8], a[8:]
        self.assertIsNone(
            np.testing.assert_array_equal(
                cu.train_test_split(a, 0.8)[0], a_train))
        self.assertIsNone(
            np.testing.assert_array_equal(
                cu.train_test_split(a, 0.8)[1], a_test))
        self.assertRaises(ValueError, cu.train_test_split, a_list, .5)
        self.assertRaises(ValueError, cu.train_test_split, a, 1.5)

    def test_mse(self):
        y_true_list = [0, 1, 2, 3, 4, 5]
        y_true = np.array(y_true_list)
        y_pred_list = [1, 2, 3, 4, 5, 6]
        y_pred = np.array(y_pred_list)
        mse = 1.0
        self.assertEqual(cu.mse(y_true, y_pred), mse)
        self.assertRaises(ValueError, cu.mse, y_true_list, y_pred_list)

    def test_rmse(self):
        y_true_list = [0, 1, 2, 3, 4, 5]
        y_true = np.array(y_true_list)
        y_pred_list = [1, 2, 3, 4, 5, 6]
        y_pred = np.array(y_pred_list)
        rmse = 1.0
        self.assertEqual(cu.rmse(y_true, y_pred), rmse)
        self.assertRaises(ValueError, cu.rmse, y_true_list, y_pred_list)
    
    def test_msle(self):
        y_true_list = [0, 1, 2, 3, 4, 5]
        y_true = np.array(y_true_list)
        y_pred_list = [1, 2, 3, 4, 5, 6]
        y_pred = np.array(y_pred_list)
        msle = 0.13906
        self.assertAlmostEqual(round(cu.msle(y_true, y_pred), 5), msle, places=4)
        self.assertRaises(ValueError, cu.msle, y_true_list, y_pred_list)
    
    def test_mae(self):
        y_true_list = [0, 1, 2, 3, 4, 5]
        y_true = np.array(y_true_list)
        y_pred_list = [1, 2, 3, 4, 5, 6]
        y_pred = np.array(y_pred_list)
        mae = 1.0
        self.assertEqual(cu.mse(y_true, y_pred), mae)
        self.assertRaises(ValueError, cu.mae, y_true_list, y_pred_list)


if __name__ == "__main__":
    unittest.main()