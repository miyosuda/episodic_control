# -*- coding: utf-8 -*-
import sys
from unittest import TestLoader
from unittest import TextTestRunner

def main():
  loader = TestLoader()
  test = loader.discover('.', pattern="*_test.py")
  runner = TextTestRunner()
  runner.run(test)  

if __name__ == '__main__':
  main()
