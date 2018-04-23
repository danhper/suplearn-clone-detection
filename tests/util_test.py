import unittest

from suplearn_clone_detection import util


class UtilTest(unittest.TestCase):
    def test_in_batch(self):
        sample_list = [8, 4, 2, 1, 9, 2, 3, 5] # 8 elements
        batches = list(util.in_batch(sample_list, batch_size=2))
        self.assertEqual(len(batches), 4)
        self.assertListEqual(batches[0], [8, 4])

        # partial batch
        sample_list.append(9)
        batches = list(util.in_batch(sample_list, batch_size=2))
        self.assertEqual(len(batches), 5)
        self.assertListEqual(batches[-1], [9])

    def test_group_by(self):
        sample_list = [(0, 1), (0, 2), (1, 2), (1, 8), (1, 4)]
        grouped = util.group_by(sample_list, key=lambda x: x[0])
        self.assertListEqual(list(grouped.keys()), [0, 1])
        self.assertListEqual(grouped[0], [(0, v) for v in [1, 2]])
        self.assertListEqual(grouped[1], [(1, v) for v in [2, 8, 4]])
