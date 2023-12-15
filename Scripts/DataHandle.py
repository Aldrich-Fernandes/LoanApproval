import csv
import random

class PreProcess:
    def __init__(self):
        # Initial Data Holders
        self.__TrainX = []
        self.__TrainY = []
        self.__featuresToRemove = ["Loan_ID", "Self_Employed", "Gender", "Education", "Married", "Dependents"]
        self.CategoricalFeatureKeys = { "Y": 1., "Urban": 1., "N": 0., "Semiurban": 0., "Rural": 2. }
        self.ScalingData = {'means': [], 'stds': []}                                # To be used when taking on user data

    def newDataset(self): # Generates a new random dataset
        # Extract data
        Dataset = DataMethod.CsvToArray(r"DataSet/HomeLoanTrain.csv")
        Dataset = self.__RemoveFeatures(Dataset)
        Dataset = self.__AdjustSkew(Dataset)                                          # Balances approved and rejected data 

        # Feature Engineering

        self.FeatureColumns = DataMethod.Transpose(Dataset)

        self.__ConvertToInteger()                                                     # Converts Categorical data (eg. male/female) into numerical

        self.__TrainY = self.FeatureColumns.pop()

        self.__Standardisation()                                                      # Scales all data to avoid large data from skewing the processes

        self.__TrainX = DataMethod.Transpose(self.FeatureColumns)

        self.__TrainX, self.__TrainY = ShuffleData(self.__TrainX, self.__TrainY)

    def __RemoveFeatures(self, Dataset):
        # Removes specified features from the dataset
        features = DataMethod.Transpose(Dataset)
        filteredFeatures = [row[1:] for row in features if row[0] not in self.__featuresToRemove]

        return DataMethod.Transpose(filteredFeatures)

    def __AdjustSkew(self, dataset):
        # Adjusts the skewness of the dataset by balancing the number of 'Y' and 'N' labels
        Ones = 0
        Zeros = 0
        NewData = []
        size = 250


        while len(NewData) != size:
            index = random.randint(0, len(dataset) - 1)
            row = dataset[index]

            if row[-1] == 'Y' and Ones != size // 2:
                NewData.append(dataset.pop(index))
                Ones += 1

            elif row[-1] == 'N' and Zeros != size // 2:
                NewData.append(dataset.pop(index))
                Zeros += 1

        return NewData

    def __ReplaceMissingVals(self):
        # Replaces missing values in each column with the most common or mean value
        Numbers = '1234567890'
        for Column in self.FeatureColumns:
            TestElement = ""
            while TestElement == "":
                TestElement = Column[random.randint(0, len(Column) - 1)]

            if TestElement[0] in Numbers and '3+' not in Column:
                FloatVals = [float(x) for x in Column if x != ""]
                ReplacementData = round(sum(FloatVals) / len(FloatVals))
            else:
                ReplacementData = max(set(Column), key=Column.count)

            for index, element in enumerate(Column):
                if element == "":
                    Column[index] = ReplacementData

    def __ConvertToInteger(self):
        # Converts categorical values to integers using a predefined mapping
        for ColumnIndex, features in enumerate(self.FeatureColumns):
            for ElementIndex, element in enumerate(features):
                try:
                    self.FeatureColumns[ColumnIndex][ElementIndex] = float(element)
                except ValueError:
                    if element not in self.CategoricalFeatureKeys.keys():
                        self.CategoricalFeatureKeys[str(element)] = sum([ord(x) for x in element]) / 16
                    self.FeatureColumns[ColumnIndex][ElementIndex] = self.CategoricalFeatureKeys[str(element)]

    def __Standardisation(self):
        # Standardizes the data by subtracting the mean and dividing by the standard deviation for each feature
        for ind, feature in enumerate(self.FeatureColumns):
            mean = sum(feature) / len(feature)
            StandardDeviation = ((sum([x**2 for x in feature]) / len(feature)) - mean**2) ** 0.5

            self.ScalingData['means'].append(mean)
            self.ScalingData['stds'].append(StandardDeviation)

            try:
                self.FeatureColumns[ind] = [(i - mean) / StandardDeviation for i in feature]
            except ZeroDivisionError:
                print(f"Mean: {mean} \nSTD: {StandardDeviation}")
                input(self.FeatureColumns[ind])

    def getData(self, split=0.2):
        # Splits the dataset into training and test sets
        NumOfTrainData = round(len(self.__TrainX) * split)
        TestX = [self.__TrainX.pop() for _ in range(NumOfTrainData)]
        TestY = [self.__TrainY.pop() for _ in range(NumOfTrainData)]
        return self.__TrainX, self.__TrainY, TestX, TestY


    def encode(self, UserData):
        # Encodes user data by standardizing and mapping categorical values
        for x, val in enumerate(UserData):
            if val in self.CategoricalFeatureKeys.keys():
                val = float(self.CategoricalFeatureKeys[val])
            else:
                val = float(val)

            UserData[x] = (val - self.ScalingData["means"][x]) / self.ScalingData["stds"][x]

        return UserData

class DataMethod:
    @staticmethod
    def CsvToArray(path): 
        # loads all data ignoring entries with missing data
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
    def DotProduct(arr1, arr2):
        if not isinstance(arr1, list):
            arr1 = [[float(arr1) for _ in range(len(arr2[0]))] for _ in range(len(arr2))]
        elif not isinstance(arr2, list):
            arr2 = [[float(arr2)] for _ in range(len(arr1))]

        arr1Shape = len(arr1), len(arr1[0])
        arr2Shape = len(arr2), len(arr2[0])

        if arr1Shape[1] != arr2Shape[0]:
            raise ValueError(f"Matrix dimensions are not compatible for dot product: {arr1Shape} and {arr2Shape}.")

        output = [[sum(a * b for a, b in zip(row, col)) for col in DataMethod.Transpose(arr2)] for row in arr1]

        return output

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

def ShuffleData(X, Y): # shuffles two lists while maintaining thier corresspronding 
    a = list(zip(X, Y))
    random.shuffle(a)
    X, Y = zip(*a)
    return list(X), list(Y)