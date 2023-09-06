import csv, random

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
        return [round(a*b, 24) for a,b in zip(arr1, arr2)]