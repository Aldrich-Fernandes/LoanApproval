from Scripts.DataHandle.PreProcess import PreProcess

import unittest

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = PreProcess()

    def test_newDataset(self):
        pass # creates a new dataset and tests the following:
                    # if the data is a 2d array of correct dimentions of only floats

    def test_DataIsStandardised(self):
        pass

    def test_GetDataSplit(self):
        pass

    def test_encode(self):
        pass