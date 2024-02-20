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


if __name__ == "__main__":
    unittest.main()
