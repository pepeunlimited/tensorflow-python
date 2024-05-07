#
#  example_test.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import unittest
from src.time_series_forecasting import dataset as ds
import os


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
        self.assertTrue(
            os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/hello_world_plot.png")
        )

    def test_hello_world_txt(self):
        """
        Test that it read hello_world.txt
        """

        self.assertEqual(ds.hello_world_txt(), "Hello, World!\n")

    def test_wind_plot(self):
        """
        Test that it can save wind velocity
        """

        ds.wind_plot()
        self.assertTrue(os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/wind_plot.png"))
        self.assertTrue(
            os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/wind_vector_plot.png")
        )

    def test_transpose(self):
        """
        Test that it can save wind velocity
        """
        df = ds.transpose()
        with open(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/dataset_transpose.txt", "a") as f:
            f.write(df.to_string(header=True, index=True))
        self.assertTrue(
            os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/dataset_transpose.txt")
        )

    def test_date_time_plot(self):
        """
        Test that it can date time plot
        """

        ds.date_time_plot()
        self.assertTrue(
            os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/date_time_signal_plot.png")
        )
        self.assertTrue(
            os.path.isfile(f"{os.getenv('TEST_UNDECLARED_OUTPUTS_DIR')}/date_time_rfft_plot.png")
        )

    def test_normalized_plot(self):
        """
        Test that it can date time plot
        """

        ds.normalize_plot()


if __name__ == "__main__":
    unittest.main()
