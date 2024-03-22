#
#  example_test.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import unittest
import src.time_series_forecasting.window_generator as tsf


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

    def test_weather_dataset(self):
        """
        Test that it can download weather_dataset
        """

        tsf.WindowGenerator().weather_dataset()

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


if __name__ == "__main__":
    unittest.main()
