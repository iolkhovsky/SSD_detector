import unittest


class Cache:

    def __init__(self, size=1):
        self.limit = int(size)
        self.data = dict()
        self.order = []
        return

    def set_size(self, sz):
        self.limit = sz
        self.__check_limit()
        return

    def add(self, key, val):
        if key in self.data.keys():
            self.__update(key)
        else:
            self.data[key] = val
            self.order.insert(0, key)
            self.__check_limit()
        return

    def try_get(self, key):
        if key in self.data.keys():
            self.__update(key)
            return self.data[key]
        else:
            return None

    def __update(self, key):
        self.order.remove(key)
        self.order.insert(0, key)
        return

    def __check_limit(self):
        while len(self.order) > self.limit:
            del self.data[self.order[-1]]
            self.order.pop()
        return

    def __str__(self):
        return "Custom Cache"

    def __len__(self):
        return len(self.order)


class TestCache(unittest.TestCase):

    def test_base(self):
        cache = Cache(3)
        in_k = ["one", "two", "three","four", "five", "six", "seven", "eight", "one", "one", "eight", "seven", "three", "three"]
        in_v = [1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 8, 7, 3, 3]
        target = [None, None, None, None, None, None, None, None, None, 1, 8, 7, None, 3]
        res = []
        for k, v in zip(in_k, in_v):
            print("Try ", k, " res: ", cache.try_get(k))
            res.append(cache.try_get(k))
            cache.add(k, v)
        self.assertEqual(res, target)
        return


if __name__ == "__main__":
    unittest.main()
