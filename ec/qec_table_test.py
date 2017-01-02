# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from ec.qec_table import QECTable

class DebugProjection(object):
  def project(self, observation):
    return observation

class TestQECTable(unittest.TestCase):
  def test_estimate(self):
    projection = DebugProjection()
    
    qec_table = QECTable(projection,
                         1,
                         2,
                         2,
                         3)

    # Hitで推定できる
    qec_table.update([0.0], 0, 0.0)
    qec = qec_table._estimate([0.0], 0)
    self.assertTrue( qec == 0.0 )

    # Hitで推定できる
    qec_table.update([1.0], 0, 1.0)
    qec = qec_table._estimate([1.0], 0)
    self.assertTrue( qec == 1.0 )

    # 2個の近傍の平均をとる
    qec = qec_table._estimate([0.1], 0)
    self.assertTrue( qec == 0.5 )

    # Hitで上書きする
    qec_table.update([0.0], 0, 1.0)
    qec = qec_table._estimate([0.0], 0)
    self.assertTrue( qec == 1.0 )

  def test_get_max_qec_action(self):
    projection = DebugProjection()
    
    qec_table = QECTable(projection,
                         1,
                         2,
                         2,
                         3)

    # action=0, R=0.0
    qec_table.update([0.0], 0, 0.0)
    # action=1, R=1.0
    qec_table.update([0.0], 1, 1.0)

    # Rが大きい値で入れたaction=1が返ってくる
    action = qec_table.get_max_qec_action([0.0])
    self.assertTrue( action == 1 )


if __name__ == '__main__':
  unittest.main()
