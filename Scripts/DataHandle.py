import csv, random

class PreProcess:
    def __init__(self, NumOfDatasets):
        #Initial DataHolders
        self.Dataset = DataMethod.CsvToArray(r"DataSet/HomeLoanTrain.csv", NumOfDatasets)
        self.CategoricalFeatureKeys = {"Y": 1, "Yes": 1, "Male": 1, "Graduate": 1, "Urban": 1, 
                                    "N": 0, "No": 0, "Female": 0, "Not Graduate": 0, "Semiurban": 0,
                                    "Rural": 2, "3+": 2}
        self.TrainX = []
        self.TrainY = []
        
        #Dealing with categorical data
        self.FeatureNames = self.Dataset.pop(0)   
        self.FeatureDict = self.CleanData()


        # Spliiting for Training data
        self.TestX, self.TestY = self.SplitData(percent=0.1)
        #self.Display()

    def Display(self):        
        print("\nTraining Data")
        for row in list(zip(self.TrainX, self.TrainY)):
            print(row)

        print("\nTesting Data")
        for row in list(zip(self.TestX, self.TestY)):
            print(row)

        Y = self.FeatureDict[-1]["Y"]
        N = self.FeatureDict[-1]["N"]
        print(f"Yes: {Y} No: {N}")

    def CleanData(self):
        FeatureColumns = DataMethod.Transpose(self.Dataset)

        FeatureColumns = self.ReplaceMissingVals(FeatureColumns)
        
        FeatureColumns = self.CreateFeatureColumns(FeatureColumns)

        self.TrainY = FeatureColumns.pop()

        self.TrainX = DataMethod.Transpose(FeatureColumns)
    
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
    
    def SplitData(self, percent=0.2): # 80-20
        NumOfTrainData = round(len(self.TrainX) * percent)
        TestX = [self.TrainX.pop() for i in range(NumOfTrainData)]
        TestY = [self.TrainY.pop() for i in range(NumOfTrainData)]
        return TestX, TestY

    def getData(self):
        return self.TrainX, self.TrainY, self.TestX, self.TestY

class DataMethod:
    @staticmethod
    def CsvToArray(path, NumOfDatasets): # loads all data then picks the a random chunk.
        table = []
        with open(path, "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                table.append(row)
        file.close()

        randomTable = [table.pop(random.randint(1, len(table)-1)) if x!=0 else table.pop(0) for x in range(NumOfDatasets+1)]
        for index, row in enumerate(randomTable): #removes loan ID from array
            randomTable[index] = row[1:]
        return randomTable

    @staticmethod
    def Transpose(array):
        return [[array[x][y] for x in range(len(array))] for y in range(len(array[0]))]
    
    @staticmethod
    def DotProduct(arr1, arr2):
        return round(sum([x*y for x,y in zip(arr1, arr2)]), 8)
    
    @staticmethod
    def Multiply(arr1, arr2):
        return [round(a*b, 8) for a,b in zip(arr1, arr2)]

