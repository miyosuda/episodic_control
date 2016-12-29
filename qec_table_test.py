# -*- coding: utf-8 -*-
import unittest

from qec_table import QECTable

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

    qec_table.update([0.0], 0, 0.0)
    qec = qec_table.estimate([0.0], 0)
    self.assertTrue( qec == 0.0 )

    qec_table.update([1.0], 0, 1.0)
    qec = qec_table.estimate([1.0], 0)
    self.assertTrue( qec == 1.0 )

    qec = qec_table.estimate([0.1], 0)
    self.assertTrue( qec == 0.5 )

    qec_table.update([0.0], 0, 1.0)
    qec = qec_table.estimate([0.0], 0)
    self.assertTrue( qec == 1.0 )

if __name__ == '__main__':
  unittest.main()
