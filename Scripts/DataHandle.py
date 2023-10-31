import csv, random

class PreProcess:
    def __init__(self, New=False):
        #Initial DataHolders
        self.__TrainX = []
        self.__TrainY = []
        self.CategoricalFeatureKeys = {"Y": 1., "Yes": 1., "Male": 1., "Graduate": 1., "Urban": 1., 
                                    "N": 0., "No": 0., "Female": 0., "Not Graduate": 0., "Semiurban": 0.,
                                    "Rural": 2., "3+": 2.}
        self.ScalingData = {'means': [],
                            'stds': []}
        if New:
            self.NewDataset()

    def NewDataset(self):

        # Extract data
        Dataset = DataMethod.CsvToArray(r"DataSet/HomeLoanTrain.csv")

        Dataset = self.AdjustSkew(Dataset)

        # Feature Engineering
        self.FeatureColumns = DataMethod.Transpose(Dataset)

        #self.ReplaceMissingVals()

        self.ConvertToInterger()

        self.__TrainY = self.FeatureColumns.pop()

        self.Standardisation()

        self.__TrainX = DataMethod.Transpose(self.FeatureColumns)

    def AdjustSkew(self, dataset):
        Ones = 0
        Zeros = 0
        index = 0
        NewDataset = []
        size = int(input("Enter the size of the dataset to use (50-600): "))

        while len(NewDataset) != size:
            index = random.randint(0, len(dataset)-1)
            row = dataset[index]
            if (row[-1] == 'Y' and Zeros > Ones) or (row[-1] == 'N' and Ones > Zeros): 
                NewDataset.append(dataset.pop(index))

            if Zeros == Ones:
                NewDataset.append(dataset.pop(index))

        return NewDataset

    def ReplaceMissingVals(self): 
        Numbers = '1234567890'
        for Column in self.FeatureColumns: # Selects the most common / mean input
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

    def ConvertToInterger(self): 
        self.CategoricalFeatureKeys = {"Y": 1., "Yes": 1., "Male": 1., "Graduate": 1., "Urban": 1., 
                                    "N": 0., "No": 0., "Female": 0., "Not Graduate": 0., "Semiurban": 0.,
                                    "Rural": 2., "3+": 2.}

        for ColumnIndex, features in enumerate(self.FeatureColumns): # for each column
            for ElementIndex, element in enumerate(features):
                try:
                    self.FeatureColumns[ColumnIndex][ElementIndex] = float(element)

                except ValueError:
                    if element not in self.CategoricalFeatureKeys.keys():
                        self.CategoricalFeatureKeys[str(element)] = sum([ord(x) for x in element]) / 16

                    self.FeatureColumns[ColumnIndex][ElementIndex] = self.CategoricalFeatureKeys[str(element)]

    def Standardisation(self): # If feature has all same val, std = 0 hence zero devision error
        for ind, feature in enumerate(self.FeatureColumns):
            mean = sum(feature) / len(feature)
            StandardDeviation = ((sum([x**2 for x in feature])/len(feature)) - mean**2)**0.5

            self.ScalingData['means'].append(mean)
            self.ScalingData['stds'].append(StandardDeviation)

            try:
                self.FeatureColumns[ind] = [float((i-mean)/StandardDeviation) for i in feature]
            except ZeroDivisionError:
                print(f"Mean: {mean} \nSTD: {StandardDeviation}")
                input(self.FeatureColumns[ind])

    def SplitData(self, percent=0.1): # 80-20
        NumOfTrainData = round(len(self.__TrainX) * percent)
        TestX = [self.__TrainX.pop() for i in range(NumOfTrainData)]
        TestY = [self.__TrainY.pop() for i in range(NumOfTrainData)]
        return TestX, TestY

    def getData(self, split=0.1):
        # Spliiting for Training data
        TestX, TestY = self.SplitData(percent=split)
        return self.__TrainX, self.__TrainY, TestX, TestY

    def Display(self):        
        print("\nTraining Data")
        for row in list(zip(self.__TrainX, self.__TrainY)):
            print(row)

    def encode(self, UserData):
        # 1. Enuerate all the data replacing matching data with those in self.categoricalfeaturekey.
        # 2. Loop through user data and standardise each of the values
        for x, val in enumerate(UserData):
            if val in self.CategoricalFeatureKeys.keys():
                val = float(self.CategoricalFeatureKeys[val])
            else:
                val = float(val)

            UserData[x] = (val - self.ScalingData["means"][x]) / self.ScalingData["stds"][x]

        return UserData


class DataMethod:
    @staticmethod
    def CsvToArray(path): # loads all data then picks the a random chunk.
        table = []
        with open(path, "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                if '' not in row:
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

        arr2Shape = [len(arr2), len(arr2[0])]
        arr1Shape = [len(arr1), len(arr1[0])]
        if arr1Shape[1] == arr2Shape[0]: # valid matrixes to multiply
            Output = []
            for rowIndex, row in enumerate(arr1):
                Output.append([])
                for column in DataMethod.Transpose(arr2):
                    Output[rowIndex].append(sum(a*b for a,b in zip(row, column)))
            return Output

        else:
            print(f"Not capable of dotting as \n  Array1:{arr1Shape}\n  Array2:{arr2Shape}")
            input()

    @staticmethod
    def Multiply(arr1, arr2):
        if type(arr1) != list:
            arr1 = [float(arr1) for x in range(len(arr2))]
        return [round(a*b, 24) for a,b in zip(arr1, arr2)]