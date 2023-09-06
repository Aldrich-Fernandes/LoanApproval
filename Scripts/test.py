import numpy as np
import csv, random
from math import log


class NeuralNetwork:        
    def train(self, mode, NumOfDatasets=300):
        # Important Values
        self.TrainX, self.TrainY, self.TestX, self.TestY = PreProcess(mode, NumOfDatasets).getData() # max 614
        self.Loss = 0.0
        self.Accuracy = 0.0

        #Create Network
        Hiddenlayer1 = Layer(11, 7, ReLU())
        Hiddenlayer2 = Layer(7, 4, ReLU())
        Outputlayer = Layer(4, 1, Sigmoid())

        BinaryLoss = BinaryCrossEntropy()

        # Training Values
        LowestLoss = 9999999
        Epochs = 2000

        BestWeight_H1 = Hiddenlayer1.weights.copy()
        BestBiases_H1 = Hiddenlayer1.biases.copy()

        BestWeight_H2 = Hiddenlayer2.weights.copy()
        BestBiases_H2 = Hiddenlayer2.biases.copy()

        BestWeight_O = Outputlayer.weights.copy()
        BestBiases_O = Outputlayer.biases.copy()

        # Epochs
        for iteration in range(Epochs):
        
            Hiddenlayer1.incrementVals()
            Hiddenlayer2.incrementVals()
            Outputlayer.incrementVals()

            Hiddenlayer1.forward(self.TrainX)
            Hiddenlayer2.forward(Hiddenlayer1.activation.outputs)
            Outputlayer.forward(Hiddenlayer2.activation.outputs)

            result = Outputlayer.activation.outputs.copy()

            self.Loss = BinaryLoss.calculate(result, self.TrainY)

            self.Accuracy = sum([1 for x,y in zip(result, self.TrainY) if round(x)==y]) / len(result)
            
            if self.Loss < LowestLoss:
                self.DisplayResults(iteration, LowestLoss)

                BestWeight_H1 = Hiddenlayer1.weights.copy()
                BestBiases_H1 = Hiddenlayer1.biases.copy()

                BestWeight_H2 = Hiddenlayer2.weights.copy()
                BestBiases_H2 = Hiddenlayer2.biases.copy()

                BestWeight_O = Outputlayer.weights.copy()
                BestBiases_O = Outputlayer.biases.copy()

                LowestLoss = self.Loss
            else:
                Hiddenlayer1.weights = BestWeight_H1.copy()
                Hiddenlayer1.biases = BestBiases_H1.copy()

                Hiddenlayer2.weights = BestWeight_H2.copy()
                Hiddenlayer2.biases = BestBiases_H2.copy()

                Outputlayer.weights = BestWeight_O.copy()
                Outputlayer.biases = BestBiases_O.copy()

            if iteration % 100 == 0:
                self.DisplayResults(iteration, LowestLoss) 

            BinaryLoss.backward(result, self.TrainY)
            Outputlayer.backward(BinaryLoss.dinputs)
            Hiddenlayer2.backward(Outputlayer.dinputs)
            Hiddenlayer1.backward(Hiddenlayer2.dinputs)
            
        # test
        Hiddenlayer1.forward(self.TestX)
        Hiddenlayer2.forward(Hiddenlayer1.activation.outputs)
        Outputlayer.forward(Hiddenlayer2.activation.outputs)

        result = Outputlayer.activation.outputs.copy()
        for x in range(len(result)):
            print(f"True: {self.TestY[x]} Predicted: {round(result[x])}")
        
        print(sum([1 for x,y in zip(result, self.TestY) if round(x)==y]) / len(result))

    def DisplayResults(self, iteration, LowestLoss):
        print(f"Iteration: {iteration} Loss: {round(LowestLoss, 5)} Accuracy: {round(self.Accuracy, 5)}\n\n")

    
class ReLU:
    def forward(self, inputs):
        self.outputs = [[0 if element < 0 else element for element in entry]
                        for entry in inputs]

    def backward(self, dvalues):
        self.dinputs = [[1 if element < 0 else 1 for element in entry] 
                        for entry in dvalues]

class Sigmoid:
    def forward(self, inputs):
        inputs = list(map(lambda z: z[0], inputs))
        self.outputs = [1 / (np.exp(-val) + 1) for val in inputs]
    
    def backward(self, dvalues):
        self.dinputs = [[(np.exp(-x) / ((1 + 2*np.exp(-x) + np.exp(-x*2))))] for x in dvalues]

# Loss
class Loss:
    def calculate(self, output, y):
        SampleLosses = self.forward(output, y)
        return SampleLosses #DataLoss

class BinaryCrossEntropy(Loss): 
    def forward(self, predictions, TrueVals):
        # Remove any 0s or 1s to avoid arithmethic errors
        for index, val in enumerate(predictions):
            if val < 0.0000001:
                predictions[index] = 0.0000001
            elif val > 0.9999999:
                predictions[index] = 0.9999999

        SampleLoss = [-(val1+val2) for val1, val2 in zip(DM.Multiply(TrueVals, [log(x) for x in predictions]),  #Probabilty of 1
                                                        DM.Multiply([1-x for x in TrueVals], [log(1-x) for x in predictions]))] # Probablity of 0
        
        SampleLoss = round(sum(SampleLoss) / len(SampleLoss), 16)

        return SampleLoss
    
    def backward(self, dvalues, TrueVals):

        for index, val in enumerate(dvalues):
            if val < 0.0000001:
                dvalues[index] = 0.0000001
            elif val > 0.9999999:
                dvalues[index] = 0.9999999

        # -(true / dvalues) + ( (1-true) / (1-dvalues))
        vals1 = [-(x/y) for x, y in zip(TrueVals, dvalues)]
        vals2 = [x/y for x, y in zip([1-i for i in TrueVals],
                                     [1-j for j in dvalues])]
        self.dinputs = [v1+v2 for v1, v2 in zip(vals1, vals2)]
    

class Layer:
    def __init__(self, NoOfInputs, NoOfNeurons, activation):
        self.__NoOfInputs = NoOfInputs
        self.__NoOfNeurons = NoOfNeurons
        self.weights = [DM.Multiply(0.01, np.random.randn(1, NoOfNeurons).tolist()[0])
                       for i in range(NoOfInputs)]
    
        self.biases = [0.0 for x in range(NoOfNeurons)]
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs.copy()

        self.output = [[a+b for a,b in zip(sample, self.biases)] for sample in DM.DotProduct(inputs, self.weights)] # add biases  -- (10x7)/ (samplesize x NOofNeurons)        

        self.activation.forward(self.output)

    def backward(self, dvalues):
        self.activation.backward(dvalues)

        dvalues = self.activation.dinputs.copy()

        self.dweights = [DM.DotProduct(DM.Transpose(self.inputs), dvalues)]
        self.dbiases = sum([x[0] for x in dvalues])

        self.dinputs = DM.DotProduct(dvalues, DM.Transpose(self.weights))

    def incrementVals(self, multiplier=0.05):
        FractionIncrease = [DM.Multiply(multiplier, np.random.randn(1, self.__NoOfNeurons).tolist()[0])
                       for sample in range(self.__NoOfInputs)]
        
        self.weights = [[a+b for a,b in zip(FractionIncrease[i], self.weights[i])] for i in range(self.__NoOfInputs)]

        self.biases = [a+b for a,b in zip(self.biases, DM.Multiply(multiplier, 
                                                                   np.random.randn(1, self.__NoOfNeurons).tolist()[0]))]

class PreProcess:
    def __init__(self, mode, NumOfDatasets):
        #Initial DataHolders
        self.NumOfDatasets = NumOfDatasets
        self.mode = mode

        self.CategoricalFeatureKeys = {"Y": 1, "Yes": 1, "Male": 1, "Graduate": 1, "Urban": 1, 
                                    "N": 0, "No": 0, "Female": 0, "Not Graduate": 0, "Semiurban": 0,
                                    "Rural": 2, "3+": 2}
        self.TrainX = []
        self.TrainY = []

        self.ChooseMode()

    def ChooseMode(self):
        if self.mode == "New":
            self.NewDataset()
        elif self.mode == "Load":
            self.LoadData()
        else:
            print("Error")

    # Generates and saves a new dataset
    def NewDataset(self):

        # Extract data
        Dataset = DataMethod.CsvToArray(r"DataSet/HomeLoanTrain.csv")
        
        Dataset = self.AdjustSkew(Dataset)
        #Dealing with categorical data

        FeatureColumns = DataMethod.Transpose(Dataset)

        FeatureColumns = self.ReplaceMissingVals(FeatureColumns)
        
        FeatureColumns = self.CreateFeatureColumns(FeatureColumns)

        self.TrainY = FeatureColumns.pop()

        self.TrainX = DataMethod.Transpose(FeatureColumns)

        # Saves cleaned data
        self.SaveData()
    
    def CreateFeatureColumns(self, FeatureColumns): #############################    Broken
        
        for ColumnIndex, features in enumerate(FeatureColumns): # for each column
            for ElementIndex, element in enumerate(features):
                try:
                    FeatureColumns[ColumnIndex][ElementIndex] = float(element)
                except ValueError:
                    if element not in self.CategoricalFeatureKeys.keys():
                        self.CategoricalFeatureKeys[str(element)] = sum([ord(x) for x in element]) / 16
                    
                    FeatureColumns[ColumnIndex][ElementIndex] = self.CategoricalFeatureKeys[str(element)]
            
        return FeatureColumns
    
    def ReplaceMissingVals(self, FeatureColumns): #############################    Broken
        Numbers = '1234567890'
        for Column in FeatureColumns: # Selects the most common / mean input
            TestElement = ""
            while TestElement == "":
                TestElement = Column[random.randint(0, len(Column)-1)]

            if TestElement[0] in Numbers and '3+' not in Column:
                FloatVals = [float(x) for x in Column if x != ""]
                ReplacementData = round(sum(FloatVals)/len(FloatVals))
            else:
                ReplacementData = max(set(Column), key = Column.count)
            
            for index, element in enumerate(Column):
                if element == "":
                    Column[index] = ReplacementData
        return FeatureColumns
    
    def SplitData(self, percent=0.1): # 80-20
        NumOfTrainData = round(len(self.TrainX) * percent)
        TestX = [self.TrainX.pop() for i in range(NumOfTrainData)]
        TestY = [self.TrainY.pop() for i in range(NumOfTrainData)]
        return TestX, TestY

    def AdjustSkew(self, dataset):
        Ones = 0
        Zeros = 0
        index = 0
        NewDataset = []

        while index < self.NumOfDatasets:
            row = dataset[index]
            if row[-1] == 'Y' and Zeros > Ones: 
                NewDataset.append(row)
            elif row[-1] == 'N' and Ones > Zeros:
                NewDataset.append(row)
            else:
                NewDataset.append(row)
            index += 1
        
        return NewDataset

    def SaveData(self):
        FileName = str(input("Save file as: "))
        TrainX = self.TrainX.copy()
        TrainY = self.TrainY.copy()

        with open(f"DataSet/{FileName}.csv", "w", newline="") as file:
            csvWriter = csv.writer(file)
            
            for index, row in enumerate(TrainX):
                row.append(TrainY[index])
                csvWriter.writerow(row)
            
            file.close()
        
        self.TrainX = [arr[:-1] for arr in self.TrainX]

    # Loads a preProccesed Dataset
    def LoadData(self):
        FileName = str(input("Load File: "))

        with open(f"DataSet/{FileName}.csv", "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                self.TrainX.append(list(map(float, row[:-1])))
                self.TrainY.append(int(row[-1]))

    # Returns Dataset for neural network
    def getData(self):
        # Spliiting for Training data
        self.TestX, self.TestY = self.SplitData(percent=0.1)

        return self.TrainX, self.TrainY, self.TestX, self.TestY

    def Display(self):        
        print("\nTraining Data")
        for row in list(zip(self.TrainX, self.TrainY)):
            print(row)
        

class DataMethod:
    @staticmethod
    def CsvToArray(path): # loads all data then picks the a random chunk.
        table = []
        with open(path, "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                table.append(row)
        file.close()

        table.pop(0) # removes feature names
        for index, row in enumerate(table): #removes loan ID from array
            table[index] = row[1:]
        return table

    @staticmethod
    def Transpose(array):
        return [[array[x][y] for x in range(len(array))] for y in range(len(array[0]))]
    
    @staticmethod
    def DotProduct(arr1, arr2): # change from vector dot product to matrix dot product
        if type(arr1) != list: 
            arr1 = [float(arr1) for i in range(len(arr2))]
        elif type(arr2) != list:
            arr2 = [[float(arr2)] for i in range(len(arr1))]

        #input(f"arr1: {arr1} \narr2: {arr2}")
        arr1Shape = [len(arr1), len(arr1[0])]
        arr2Shape = [len(arr2), len(arr2[0])]
        if arr1Shape[1] == arr2Shape[0]: # valid matrixes to multiply
            #print(arr1Shape, arr2Shape)
            Output = []
            for rowIndex, row in enumerate(arr1):
                Output.append([])
                for column in DataMethod.Transpose(arr2):
                    Output[rowIndex].append(sum(a*b for a,b in zip(row, column)))
            
            #input([len(Output), len(Output[0])])
            return Output

        else:
            print(f"Not capable of dotting as \n  Array1:{arr1Shape}\n  Array2:{arr2Shape}")
            input()
    
    @staticmethod
    def Multiply(arr1, arr2):
        if type(arr1) != list:
            arr1 = [float(arr1) for x in range(len(arr2))]
        return [round(a*b, 16) for a,b in zip(arr1, arr2)]

Option = int(input("Do you want to train a new (1) or use a old (2) dataset:"))
if Option == 1:
    mode = "New"
    NoOfSamples = int(input("How many samples (Max: 600): "))
elif Option == 2:
    mode = "Load"
    NoOfSamples = 600

NeuralNetwork().train(mode, NoOfSamples)