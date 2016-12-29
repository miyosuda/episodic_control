# -*- coding: utf-8 -*-
import unittest

from knn import KNN

class TestKNN(unittest.TestCase):
  def test_peek(self):
    knn = KNN(3, 1)

    # 4回エントリを入れる
    knn.update([0.0], 0.0)
    knn.update([1.0], 1.0)
    knn.update([2.0], 2.0)
    knn.update([3.0], 3.0)

    # エントリ数は3つ
    self.assertTrue( knn._current_capacity == 3 )

    # Hitする場合
    peeked = knn.peek([2.0])
    self.assertTrue( peeked == 2.0 )

    # Hitしない場合
    peeked = knn.peek([2.1])
    self.assertTrue( peeked == None )

    # 一番最初のエントリはLRUのおかげで消えている
    peeked = knn.peek([0.0])
    self.assertTrue( peeked == None )

  def test_update(self):
    knn = KNN(3, 1)
    knn.update([0.0], 0.0)
    knn.update([1.0], 1.0)
    knn.update([2.0], 2.0)

    # Hitの確認
    peeked = knn.peek([0.0])
    self.assertTrue( peeked == 0.0 )

    # 上書きする
    knn.update([0.0], 0.5)

    # エントリは3つのまま
    self.assertTrue( knn._current_capacity == 3 )

    # Hitする値が変わっているのを確認
    peeked = knn.peek([0.0])
    self.assertTrue( peeked == 0.5 )

  def test_knn_value(self):
    knn = KNN(3, 1)
    knn.update([0.0], 0.0)
    knn.update([1.0], 1.0)
    knn.update([2.0], 2.0)

    # 1.0と2.0の平均の1.5が返ってくるはず
    q = knn.knn_value([1.2], 2)
    self.assertTrue( q == 1.5 )

    # K個未満のエントリしかない場合のテスト
    knn = KNN(3, 1)
    # エントリ0個の場合 (デフォルト値の0を返している) # TODO: Noneにすべき？
    q = knn.knn_value([0.0], 2)
    self.assertTrue( q == 0.0 )

    # エントリ1個の場合 (デフォルト値の0を返している) # TODO: Noneにすべき？
    knn.update([1.0], 1.0)
    q = knn.knn_value([1.0], 2)
    self.assertTrue( q == 0.0 )

    # エントリ1個の場合 (平均が正常に取れる)
    knn.update([2.0], 2.0)
    q = knn.knn_value([1.0], 2)
    self.assertTrue( q == 1.5 )


if __name__ == '__main__':
  unittest.main()
