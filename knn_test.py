# -*- coding: utf-8 -*-
import unittest

from knn import KNN

class TestKNN(unittest.TestCase):
  def test_peek(self):
    knn = KNN(3, 1)
    knn.update([0.0], 0.0)
    knn.update([1.0], 1.0)
    knn.update([2.0], 2.0)
    knn.update([3.0], 3.0)

    self.assertTrue( knn._current_capacity == 3 )

    peeked = knn.peek([2.0])
    self.assertTrue( peeked == 2.0 )

    peeked = knn.peek([2.1])
    self.assertTrue( peeked == None )

    peeked = knn.peek([0.0])
    self.assertTrue( peeked == None )

  def test_update(self):
    knn = KNN(3, 1)
    knn.update([0.0], 0.0)
    knn.update([1.0], 1.0)
    knn.update([2.0], 2.0)

    peeked = knn.peek([0.0])
    self.assertTrue( peeked == 0.0 )

    knn.update([0.0], 0.5)

    self.assertTrue( knn._current_capacity == 3 )

    peeked = knn.peek([0.0])
    self.assertTrue( peeked == 0.5 )

  def test_knn_value(self):
    knn = KNN(3, 1)
    knn.update([0.0], 0.0)
    knn.update([1.0], 1.0)
    knn.update([2.0], 2.0)

    q = knn.knn_value([1.2], 2)
    self.assertTrue( q == 1.5 )


if __name__ == '__main__':
  unittest.main()
