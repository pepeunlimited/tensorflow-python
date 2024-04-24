#
#  example_test.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import unittest
from src.time_series_forecasting import dataset as ds


class TestDataset(unittest.TestCase):

    def test_weather_dataset(self):
        """
        Test that it can download weather_dataset
        """

        train_df, val_df, test_df = ds.weather_dataset()

    def test_hello_world_plot(self):
        """
        Test that it can save hello_world_plot.png
        """

        ds.hello_world_plot()

    def test_hello_world_txt(self):
        """
        Test that it read data.txt
        """

        self.assertEqual(ds.hello_world_txt(), "Hello, World!\n")


if __name__ == "__main__":
    unittest.main()
