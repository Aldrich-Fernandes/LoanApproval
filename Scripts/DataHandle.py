import csv, random

class PreProcess:
    def __init__(self, NumOfDatasets, path=r"DataSet\HomeLoanTrain.csv"):
        self.Dataset = DataMethod.CsvToArray(path, NumOfDatasets)
        self.DataX = []
        self.DataY = []
        
        self.FeatureNames = self.Dataset.pop(0)    
        self.FeatureDict = self.CleanData() 

        self.Display()

    def Display(self):
        for row in list(zip(self.DataX, self.DataY)):
            print(row)

    def CleanData(self):
        FeatureColumns = DataMethod.Transpose(self.Dataset)

        FeatureColumns = self.ReplaceMissingVals(FeatureColumns)
        
        FeatureColumns, CategoricalFeatureKeys = self.CreateFeatureColumns(FeatureColumns)

        self.DataY = FeatureColumns.pop()

        self.DataX = DataMethod.Transpose(FeatureColumns)
        return CategoricalFeatureKeys
    
    def CreateFeatureColumns(self, FeatureColumns):
        CategoricalFeatureKeys = []
        for index, features in enumerate(FeatureColumns): # for each column
            try: # For decimals
                FeatureColumns[index] = list(map(float, features)) # turn string to float
            except: # for strings  
                FeatureDict = {}
                val = 0
                for RowIndex, element in enumerate(features):
                    if element not in FeatureDict.keys():
                        FeatureDict[element] = val 
                        val += 1 
                    features[RowIndex] = float(FeatureDict[element])
                CategoricalFeatureKeys.append(FeatureDict)
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
        TransposedArray = []
        for y in range(len(array[0])):
            TransposedArray.append([array[x][y] for x in range(len(array))])
        
        return TransposedArray
    