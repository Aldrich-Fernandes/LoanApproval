import csv, random
import matplotlib.pyplot as plt

class DataPrep:
    def __init__(self, NumOfDatasets, path=r"DataSet\loan_sanction_train.csv"):
        self.Dataset = DataMethod.CsvToArray(path, NumOfDatasets)
        self.FeatureNames = self.Dataset.pop(0)
        self.FeatureKeys, self.DataY = self.CleanData() # DataY = LoanStatus
        self.Display()

    def Display(self):
        print("\n",self.FeatureNames)
        for row in self.Dataset:
            print(row)

    def CleanData(self):
        FeatureColumns = DataMethod.Transpose(self.Dataset)

        FeatureColumns = self.ReplaceMissingVals(FeatureColumns)
        
        FeatureColumns, CategoricalFeatureKeys = self.CreateFeatureColumns(FeatureColumns)

        DataY = FeatureColumns.pop()
        self.FeatureNames.pop()

        self.Dataset = DataMethod.Transpose(FeatureColumns)
        return CategoricalFeatureKeys, DataY

    def ShuffleOrder(self):
        random.shuffle(self.Dataset)
    
    def CreateFeatureColumns(self, FeatureColumns):
        CategoricalFeatureKeys = []
        for index, features in enumerate(FeatureColumns): # for each column
            try: # For intergers
                FeatureColumns[index] = list(map(int, features)) # turn string to int
            except: # for strings  
                FeatureKeys = {}
                val = 0
                for index, element in enumerate(features):
                    if element not in FeatureKeys.keys():
                        FeatureKeys[element] = val 
                        val += 1 
                    features[index] = int(FeatureKeys[element])
                CategoricalFeatureKeys.append(FeatureKeys)
        return FeatureColumns, CategoricalFeatureKeys
    
    def ReplaceMissingVals(self, FeatureColumns):
        for Column in FeatureColumns:
            if type(Column) == int:
                ReplacementData = sum(Column)/len(Column)
            else:
                ReplacementData = max(set(Column), key = Column.count)

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
    
    def ArrayToCsv(path, Data):
        raise NotImplementedError

    @staticmethod
    def Transpose(array):
        TransposedArray = []
        for y in range(len(array[0])):
            TransposedArray.append([array[x][y] for x in range(len(array))])
        
        return TransposedArray