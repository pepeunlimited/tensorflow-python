#
#  example_test.py
#
#  Copyright 2024 Pepe Unlimited
#  Licensed under the MIT license, see associated LICENSE file for terms.
#  See AUTHORS file for the list of project authors.
#

import unittest
import src.lib.haberdasher as haberdasher


class TestHaberdasher(unittest.TestCase):
    def test_make_hat(self):
        """
        Test that it can return correct hat
        """
        self.assertEqual(haberdasher.make_hat(), "inches: 12, color: invisible, name: bowler")


if __name__ == "__main__":
    unittest.main()
