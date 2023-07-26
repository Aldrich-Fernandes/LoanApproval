import csv, random

class PreProcess:
    def __init__(self, NumOfDatasets, path=r"DataSet/HomeLoanTrain.csv"):
        #Initial DataHolders
        self.Dataset = DataMethod.CsvToArray(path, NumOfDatasets)
        self.TrainX = []
        self.TrainY = []
        
        #Dealing with categorical data
        self.FeatureNames = self.Dataset.pop(0)   
        self.FeatureDict = self.CleanData()


        # Spliiting for Training data
        self.TestX, self.TestY = self.SplitData()

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
        
        FeatureColumns, CategoricalFeatureKeys = self.CreateFeatureColumns(FeatureColumns)

        self.TrainY = FeatureColumns.pop()

        self.TrainX = DataMethod.Transpose(FeatureColumns)
        return CategoricalFeatureKeys
    
    def CreateFeatureColumns(self, FeatureColumns):
        CategoricalFeatureKeys = {"Y" or "Yes" or "Male" or "Graduate": 1, 
                                  "N" or "No" or "Female" or "Not Graduate": 0}
        for ColumnIndex, features in enumerate(FeatureColumns): # for each column
            for ElementIndex, element in enumerate(features):
                try:
                    FeatureColumns[ColumnIndex][ElementIndex] = float(element)
                except ValueError:
                    if element not in CategoricalFeatureKeys.keys():
                        CategoricalFeatureKeys[str(element)] = sum([ord(x) for x in element]) / 16
                    
                    FeatureColumns[ColumnIndex][ElementIndex] = CategoricalFeatureKeys[str(element)]
        return FeatureColumns, CategoricalFeatureKeys
    
    def ReplaceMissingVals(self, FeatureColumns):

        for Column in FeatureColumns: # Selects the most common / mean input
            try:
                Column = list(map(float, Column))
                ReplacementData = sum(Column)/len(Column)                    
            except: 
                ReplacementData = max(set(Column), key = Column.count)

            #Updates the missing vals in the column
            for index, element in enumerate(Column):
                if element == "":
                    Column[index] = ReplacementData
            
        return FeatureColumns
    
    def SplitData(self): # 80-20
        NumOfTrainData = round(len(self.TrainX) * 0.2)
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
