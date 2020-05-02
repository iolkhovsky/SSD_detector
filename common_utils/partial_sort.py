from heapq import heappush, heappop
import unittest


class PartialSorter:

    def __init__(self, max_sz):
        self.data = []
        self.max_sz = max_sz
        self.it_idx = 0
        return

    def push(self, element):
        heappush(self.data, element)
        self.__shrink()
        return

    def get_data(self):
        return self.data

    def __shrink(self):
        while len(self.data) > self.max_sz:
            heappop(self.data)
        return

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "Partial sorter"

    def __next__(self):
        out = self.it_idx
        if out > len(self.data) - 1:
            raise StopIteration
        self.it_idx += 1
        return out

    def __iter__(self):
        return self


class TestPartialSorter(unittest.TestCase):

    def test_base(self):
        sorter = PartialSorter(3)
        test_data = [-1, 2, 15, 78, 24, -15, 5, 26, 17, 21, 8, 64]
        for i in test_data:
            sorter.push(i)
        self.assertEqual(set(sorter.get_data()), set([64, 78, 26]))

    def test_tuples(self):
        sorter = PartialSorter(2)
        test_data = [(-1, "asd"), (2, "there"), (15, "is"), (78, "another"), (24, "way"), (-15, "to"), (5, "solve"),
                     (26, "the"), (17, "problem"), (21, "but"), (8, "this"), (64, "one"), (0, "better")]
        for i in test_data:
            sorter.push(i)
        self.assertEqual(set(sorter.get_data()), set([(64, "one"), (78, "another")]))


if __name__ == "__main__":
    unittest.main()
