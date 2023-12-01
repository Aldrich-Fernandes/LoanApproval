import csv, random

class PreProcess:
    def __init__(self, New=False):
        #Initial DataHolders
        self.__TrainX = []
        self.__TrainY = []
        self.__featuresToRemove = ["Loan_ID"
                                   , "Self_Employed", "Gender", "Education", "Married", "Dependents"
                                   ]
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
        Dataset = self.RemoveFeatures(Dataset)
        Dataset = self.AdjustSkew(Dataset, samplesize=100)

        # Feature Engineering
        self.FeatureColumns = DataMethod.Transpose(Dataset)

        #self.ReplaceMissingVals()

        self.ConvertToInteger()

        self.__TrainY = self.FeatureColumns.pop()

        self.Standardisation()

        self.__TrainX = DataMethod.Transpose(self.FeatureColumns)

        self.__TrainX, self.__TrainY = ShuffleData(self.__TrainX, self.__TrainY)

    def RemoveFeatures(self, Dataset):
        features = DataMethod.Transpose(Dataset)
        filteredFeatures = []
        for row in features:
            if row[0] not in self.__featuresToRemove:
                filteredFeatures.append(row[1:])
                
        return DataMethod.Transpose(filteredFeatures)

    def AdjustSkew(self, dataset, samplesize):
        Ones = 0
        Zeros = 0
        NewDataset = []
        size = samplesize 

        while len(NewDataset) != size:
            index = random.randint(0, len(dataset)-1)
            row = dataset[index]
            
            if row[-1] == 'Y' and Ones != size // 2: 
                NewDataset.append(dataset.pop(index))
                Ones += 1

            elif row[-1] == 'N' and Zeros != size // 2:
                NewDataset.append(dataset.pop(index))
                Zeros += 1

        return NewDataset

    def ReplaceMissingVals(self):  # needed??
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

    def ConvertToInteger(self): 
        self.CategoricalFeatureKeys = {"Y": 1., "Yes": 1., "Male": 1., "Graduate": 1., "Urban": 1., 
                                    "N": 0., "No": 0., "Female": 0., "Not Graduate": 0., "Semiurban": 0.,
                                    "Rural": 2., "3+": 2.}

        for ColumnIndex, features in enumerate(self.FeatureColumns): # for each column
            for ElementIndex, element in enumerate(features):
                try:
                    self.FeatureColumns[ColumnIndex][ElementIndex] = float(element)

                except ValueError:
                    if element not in self.CategoricalFeatureKeys.keys(): # use abs()
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
            except ZeroDivisionError: # If feature has all same val, std = 0 hence zero devision error
                print(f"Mean: {mean} \nSTD: {StandardDeviation}")
                input(self.FeatureColumns[ind])

    def getData(self, split=0.2):
        # Spliiting for Training data
        NumOfTrainData = round(len(self.__TrainX) * split)
        TestX = [self.__TrainX.pop() for _ in range(NumOfTrainData)]
        TestY = [self.__TrainY.pop() for _ in range(NumOfTrainData)]
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

        return table

    @staticmethod
    def Transpose(array):
        return [[array[x][y] for x in range(len(array))] for y in range(len(array[0]))]

    @staticmethod
    def DotProduct(arr1, arr2): # change from vector dot product to matrix dot product
        if not isinstance(arr1, list): 
            arr1 = [[float(arr1) for i in range(len(arr2))] for j in range(len(arr2[0]))] # if arr2 = (2,3) creates arr = (3, 2)
        elif type(arr2) != list:
            arr2 = [[float(arr2)] for i in range(len(arr1))]

        arr1Shape = len(arr1), len(arr1[0])
        arr2Shape = len(arr2), len(arr2[0])

        if arr1Shape[1] != arr2Shape[0]:
            raise ValueError(f"Matrix dimensions are not compatible for dot product {arr1Shape} and {arr2Shape}.")

        if arr1Shape[1] == arr2Shape[0]: # valid matrixes to multiply
            Output = [[sum(a*b for a,b in zip(row, column)) for column in DataMethod.Transpose(arr2)] for row in arr1]

            return Output

        else:
            print(f"Not capable of dotting as \n  Array1:{arr1Shape}\n  Array2:{arr2Shape}")
            input()

    @staticmethod
    def Multiply(arr1, arr2): # dimensions of arr1 Must be <= dimensions of arr2
        if not isinstance(arr1, list): # Ensures dimentions are atleast 1 dimensional
            if isinstance(arr2[0], list): # uses inside lenght of a row
                arr1 = [float(arr1) for _ in range(len(arr2[0]))]
            else:
                arr1 = [float(arr1) for _ in range(len(arr2))]

        if isinstance(arr2[0], list):  # Check if arr2 is a 2D array
            if isinstance(arr1[0], list):  # For 2d x 2d
                return [DataMethod.Multiply(row1, row2) for row1, row2 in zip(arr1, arr2)]
            else: # For 1d x 2d
                return [DataMethod.Multiply(arr1, row) for row in arr2]
        else: # For 1d x 1d
            return [a * b for a, b in zip(arr1, arr2)]

def ShuffleData(X, Y):
    a = list(zip(X, Y))
    random.shuffle(a)
    X, Y = zip(*a)
    return list(X), list(Y)