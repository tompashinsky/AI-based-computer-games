import unittest

def is_prime(number):
    if number < 2:
        return False
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return False
    return True


class Test(unittest.TestCase):

    def test(self):
        self.assertTrue(is_prime(10), False)


if __name__ == '__main__':
    unittest.main()