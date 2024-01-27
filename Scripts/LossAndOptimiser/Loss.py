from DataHandle import DataMethod as DM

from math import log


'''
Loss 

Measures how well the model performed by comparing the true and predicted values

The algorithm doesn't utilise the calculated loss value directly. It is use to visualise if the model is
improving when training and identifying what is impacting the model and how much does it. 

'''
class BinaryCrossEntropy:
    def __init__(self, regStr=0.0):
        self.__sampleLoss = 0.0                     # Measure of how far the prediction is from the true value
        self.__regLoss = 0.0                        # Additional term to sampleLoss to deter large weights
  
        self.__regStr = regStr                      # How strongly to penalise the model for large weights

    # Calculates how far the predicted values are from the true values
    def forward(self, predictions, TrueVals):
        predictions = clipEdges(predictions)

        # Formula used: -(true * log(Predicted) + (1 - true) * log(1 - Predicted))
        sampleLoss = [-((tVal * log(pVal)) + ((1 - tVal) * log(1 - pVal))) 
                      for tVal, pVal in zip(TrueVals, predictions)]

        self.__sampleLoss = sum(sampleLoss) / len(sampleLoss)   # Average of all samples

    # Gradient of what impacted the result the most
    def backward(self, predicted, TrueVals): 
        predicted = clipEdges(predicted)
        
        # Derivative of formula above used: (PredictVal - Tval) / ((1-PredictVal) * PredictVal)
        self.dinputs = [(PredictVal - Tval) / ((1-PredictVal) * PredictVal) 
                        for Tval, PredictVal in zip(TrueVals, predicted)]

    # Used to change the hyperparameter when training a new model
    def updateRegStr(self, regStr):
        self.__regStr = regStr

    # L2 regularisation foumula: 0.5 * regStr * SumOfSquaredWeights
    def calcRegularisationLoss(self, layerWeights):
        if self.__regStr != 0:
            weightSqrSum = sum([sum(x) for x in DM.Multiply(layerWeights, layerWeights)])

            self.__regLoss = 0.5 * self.__regStr * weightSqrSum

    # Returns total loss when called.
    def getLoss(self):
        return self.__sampleLoss + self.__regLoss
    
# Replaces any 0s or 1s to avoid arithmetic errors
def clipEdges(list, scale=1e-7):
    for index, val in enumerate(list):
        if val < scale:
            list[index] = scale
        elif val > 1 - (scale):
            list[index] = 1 - (scale)
    return list