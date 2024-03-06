import unittest
from Scripts.DataHandle import Preprocess

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = Preprocess()
        self.processor.newDataset()
        self.TrainX = self.processor._TrainX

    def test_newDataset(self):
        # Check if the data is a 2D array
        self.assertTrue(isinstance(self.TrainX, list))
        for row in self.TrainX:
            self.assertTrue(isinstance(row, list))
            for val in row:         # Check if the data consists of floats
                self.assertTrue(isinstance(val, float))
    
    def test_DataIsStandardised(self):
        # Check if the mean of the dataset is approximately 0
        sumOfFeatures = sum([sum(feature) for feature in self.TrainX])
        totalElements = (len(self.TrainX) * len(self.TrainX[0]))
        mean = sumOfFeatures / totalElements
        self.assertAlmostEqual(mean, 0.0, delta=0.1)

        # Check if the standard deviation of the dataset is approximately 1
        squared_sum = sum([sum([x**2 for x in feature]) for feature in self.TrainX])
        std_dev = ((squared_sum / (len(self.TrainX) * len(self.TrainX[0]))) - mean**2) ** 0.5
        self.assertAlmostEqual(std_dev, 1.0, delta=0.1)

    def test_GetDataSplit(self):
        train_X, train_Y, test_X, test_Y = self.processor.getData()

        # Check if split data has correct dimensions
        self.assertEqual(len(train_X)/ len(train_Y), len(test_X)/ len(test_Y))
    
    def test_encode(self):
        # Encode sample user data
        encoded_data = self.processor.encode([9732, 6543, 76, 180, "Yes", "Semiurban"])

        # Check if encoded data has correct dimensions
        self.assertEqual(len(encoded_data), len(self.TrainX[0]))

        # Check if encoded data consists of floats
        for val in encoded_data:
            self.assertTrue(isinstance(val, float))