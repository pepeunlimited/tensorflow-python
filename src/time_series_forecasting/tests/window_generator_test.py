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
import os


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
            wg.input_indices.tolist(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        )
        self.assertEqual(wg.label_start, 47)
        self.assertListEqual(
            wg.label_indices.tolist(),
            [47],
        )
        self.assertListEqual(wg.label_columns, [])

    def test_1h_into_future_given_6h_history(self):
        """
        Test that it can do calculate correct values
        """

        wg = tsf.WindowGenerator(input_width=6, label_width=1, shift=1)
        self.assertEqual(wg.total_window_size, 7)
        self.assertListEqual(
            wg.input_indices.tolist(),
            [0, 1, 2, 3, 4, 5],
        )
        self.assertEqual(wg.label_start, 6)
        self.assertListEqual(
            wg.label_indices.tolist(),
            [6],
        )
        self.assertListEqual(wg.label_columns, [])

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

    def test_input_label_parameters(self):
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

        self.assertListEqual(wg.label_columns, ["T (degC)"])
        self.assertDictEqual(wg.label_columns_indices, {"T (degC)": 0})
        self.assertDictEqual(
            wg.column_indices,
            {
                "p (mbar)": 0,
                "T (degC)": 1,
                "Tpot (K)": 2,
                "Tdew (degC)": 3,
                "rh (%)": 4,
                "VPmax (mbar)": 5,
                "VPact (mbar)": 6,
                "VPdef (mbar)": 7,
                "sh (g/kg)": 8,
                "H2OC (mmol/mol)": 9,
                "rho (g/m**3)": 10,
                "Wx": 11,
                "Wy": 12,
                "max Wx": 13,
                "max Wy": 14,
                "Day sin": 15,
                "Day cos": 16,
                "Year sin": 17,
                "Year cos": 18,
            },
        )
        self.assertEqual(wg.input_slice, slice(0, 6, None))
        self.assertListEqual(wg.input_indices.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(wg.label_start, 6)
        self.assertEqual(wg.label_slice, slice(6, None, None))
        self.assertListEqual(wg.label_indices.tolist(), [6])

    def test_plot(self):
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
        wg.plot(inputs=inputs, labels=labels)
        wg.plot(inputs=inputs, labels=labels, plot_col="p (mbar)")
        self.assertTrue(
            os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/TdegC_plot.png")
        )
        self.assertTrue(
            os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/pmbar_plot.png")
        )


if __name__ == "__main__":
    unittest.main()
