# -*- coding: utf-8 -*-
import sys

# This test loader doesn't work with bazel environment
"""
from unittest import TestLoader
from unittest import TextTestRunner

def main():
  loader = TestLoader()
  test = loader.discover('.', pattern="*_test.py")
  runner = TextTestRunner()
  runner.run(test)
"""

import unittest
import ec.knn_test
import ec.qec_table_test
import environment.environment_test
import projection.vae_test

def get_suite():
  suite = unittest.TestSuite()

  suite.addTest(unittest.makeSuite(ec.knn_test.TestKNN))
  suite.addTest(unittest.makeSuite(ec.qec_table_test.TestQECTable))
  suite.addTest(unittest.makeSuite(environment.environment_test.TestGameEnvironment))
  suite.addTest(unittest.makeSuite(projection.vae_test.TestVAE))
  
  return suite

def main():
  suite = get_suite()
  unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
  main()
