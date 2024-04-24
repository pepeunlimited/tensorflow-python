#
#  example_test.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import unittest
import tensorflow as tf
import pandas as pd
import numpy as np
from src.time_series_forecasting import window_generator as tsf
from src.time_series_forecasting import dataset as ds


class TestWindowGenerator(unittest.TestCase):
    def test_hello_world(self):
        """
        Test that it can print hello world
        """

        self.assertEqual(tsf.WindowGenerator().hello_world, "Hello, World!")

    def test_printer(self):
        """
        Test that it can print
        """

        tsf.WindowGenerator().printer()

    def test_24h_into_future_given_24h_history(self):
        """
        Test that it can do calculate correct values
        """

        wg = tsf.WindowGenerator(input_width=24, label_width=1, shift=24)
        self.assertEqual(wg.total_window_size, 48)
        self.assertListEqual(
            wg.input_slices.tolist(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        )
        self.assertEqual(wg.label_start, 47)
        self.assertListEqual(
            wg.label_indices.tolist(),
            [47],
        )

    def test_1h_into_future_given_6h_history(self):
        """
        Test that it can do calculate correct values
        """

        wg = tsf.WindowGenerator(input_width=6, label_width=1, shift=1)
        self.assertEqual(wg.total_window_size, 7)
        self.assertListEqual(
            wg.input_slices.tolist(),
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(wg.label_start, 6)
        self.assertListEqual(
            wg.label_indices.tolist(),
            [6],
        )

    def test_stack_3_slices_the_length_of_the_total_window(self):
        """
        Test that it can do calculate correct values
        """

        # data = {"something": [1, 2], "another": [4, 5]}
        # train_df = pd.DataFrame(data)
        # example_window0 = tf.stack([np.array(train_df[: wg.total_window_size])])
        # print(example_window0)
        # wg.split_window(example_window0)

        train_df, val_df, test_df = ds.weather_dataset()

        wg = tsf.WindowGenerator(
            input_width=6,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["T (degC)"],
        )

        # print(repr(wg))

        example_window1: tf.Tensor = tf.stack(
            [
                np.array(wg.train_df[: wg.total_window_size]),
                np.array(wg.train_df[100 : 100 + wg.total_window_size]),
                np.array(wg.train_df[200 : 200 + wg.total_window_size]),
            ]
        )

        # The code above took a batch of three 7-time step windows with 19 features at each time step.
        # It splits them into a batch of 6-time step 19-feature inputs, and a 1-time step 1-feature label.
        # The label only has one feature because the WindowGenerator was initialized with label_columns=['T (degC)'].
        # Initially, this tutorial will build models that predict single output labels.

        inputs, labels = wg.split_window(example_window1)
        self.assertListEqual(np.array(example_window1.shape).tolist(), [3, 7, 19])
        self.assertListEqual(np.array(inputs.shape).tolist(), [3, 6, 19])
        self.assertListEqual(np.array(labels.shape).tolist(), [3, 1, 1])

    def test_not_implemented(self):
        self.assertRaises(NotImplementedError, tsf.WindowGenerator().not_implemented)


if __name__ == "__main__":
    unittest.main()
