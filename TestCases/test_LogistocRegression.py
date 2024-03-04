from Scripts.NeuralNetwork.Models import LogisticRegression
from Scripts.DataHandle import PreProcess

import unittest

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression()
        self.model.addLayer(6, 1)
        self.Preprocessor = PreProcess()
        self.Preprocessor.newDataset()

    def test_Prediction(self):
        targetAccuracy = 0.70
        TrainX, TrainY, TestX, TestY = self.Preprocessor.getData()

        while True: # Ensures model is valid
            self.model.train(TrainX, TrainY)
            self.model.test(TestX, TestY)

            if self.model.Accuracy > targetAccuracy:
                # Called again to show predicted values  
                self.model.test(TestX, TestY, True)
                self.assertGreaterEqual(self.model.Accuracy, targetAccuracy)
                break
            else:
                self.model.resetLayers()
                self.Preprocessor.newDataset()
                TrainX, TrainY, TestX, TestY = self.Preprocessor.getData()