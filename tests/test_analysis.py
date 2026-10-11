import unittest
from analysis import checks


class Controls(unittest.TestCase):
    def test_controls(self):
        checks()


if __name__ == "__main__":
    unittest.main()
