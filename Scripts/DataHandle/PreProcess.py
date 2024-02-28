import random

from .DataMethods import DataMethod

'''
Raw Data cannot be processed by the model. Therefore both training and userdata have to be adjusted prior to 
training or prediction

'''
class PreProcess:
    def __init__(self, path=r"DataSet//HomeLoanTrain.csv"):
        # Initial data holders for training data
        self.__TrainX = []
        self.__TrainY = []


        # Data specific to the training data used
        self.__featuresToRemove = ["Loan_ID", "Self_Employed", "Gender", "Education", "Married", "Dependents"]
        self.CategoricalFeatureKeys = {"Y": 1., "Yes": 1., "Male": 1., "Graduate": 1., "Urban": 1., 
                                    "N": 0., "No": 0., "Female": 0., "Not Graduate": 0., "Semiurban": 0.,
                                    "Rural": 2., "3+": 2.}
        self.ScalingData = {'means': [], 'stds': []}
        self.__path = path

    # Generates a new random dataset
    def newDataset(self):
        # Extract data from file and selecting valid entries/features
        Dataset = DataMethod.CsvToArray(self.__path)
        Dataset = self.__RemoveFeatures(Dataset)
        Dataset = self.__AdjustSkew(Dataset) 

        # Feature Engineering - Making data usable for the model

        self.FeatureColumns = DataMethod.Transpose(Dataset)

        self.__ConvertToInteger()

        self.__TrainY = self.FeatureColumns.pop()

        self.__Standardisation()

        self.__TrainX = DataMethod.Transpose(self.FeatureColumns)

        self.__TrainX, self.__TrainY = DataMethod.ShuffleData(self.__TrainX, self.__TrainY)

    # Removes specified features (Attribute) from the dataset
    def __RemoveFeatures(self, Dataset):
        features = DataMethod.Transpose(Dataset)
        filteredFeatures = [row[1:] for row in features if row[0] not in self.__featuresToRemove]
        return DataMethod.Transpose(filteredFeatures)

    # Fixes skew of the dataset by balancing the number of 'Y' and 'N' labels
    def __AdjustSkew(self, dataset):
        Trues = 0           # No. of approved applications
        Falses = 0          # No. of unsuccessful applications
        NewData = []
        size = 250          # Number of entries to utilise

        while len(NewData) != size:
            index = random.randint(0, len(dataset) - 1)
            row = dataset[index]
            if row[-1] == 'Y' and Trues != size // 2:
                NewData.append(dataset.pop(index))
                Trues += 1
            elif row[-1] == 'N' and Falses != size // 2:
                NewData.append(dataset.pop(index))
                Falses += 1
        return NewData

    # Converts categorical values (such as strings) to integers using a predefined mapping
    def __ConvertToInteger(self):
        for ColumnIndex, features in enumerate(self.FeatureColumns):
            for ElementIndex, element in enumerate(features):
                try:                        # For data that is already numerical
                    self.FeatureColumns[ColumnIndex][ElementIndex] = float(element)
                except ValueError:          # For data that is categorical
                    if element not in self.CategoricalFeatureKeys.keys():
                        self.CategoricalFeatureKeys[str(element)] = sum([ord(x) for x in element]) / 16
                    self.FeatureColumns[ColumnIndex][ElementIndex] = self.CategoricalFeatureKeys[str(element)]

    # Z-score normalisation formula: (data - mean) / standard deviation
    # Improves interpretability and model performance by removing scale difference between features 
    def __Standardisation(self):
        for ind, feature in enumerate(self.FeatureColumns):
            mean = sum(feature) / len(feature)
            StandardDeviation = ((sum([x**2 for x in feature]) / len(feature)) - mean**2) ** 0.5

            # To use when scale userdata properly for the dataset
            self.ScalingData['means'].append(mean)
            self.ScalingData['stds'].append(StandardDeviation)

            # Applying the standardisation
            try:
                self.FeatureColumns[ind] = [(i - mean) / StandardDeviation for i in feature]
            except ZeroDivisionError:
                print(f"Mean: {mean} \nSTD: {StandardDeviation}")
                input(self.FeatureColumns[ind])

    # Splits the dataset into training and test sets and returns the data
    def getData(self, split=0.2): # default: 80-20 split
        NumOfTrainData = round(len(self.__TrainX) * split)
        TestX = [self.__TrainX.pop() for _ in range(NumOfTrainData)]
        TestY = [self.__TrainY.pop() for _ in range(NumOfTrainData)]
        return self.__TrainX, self.__TrainY, TestX, TestY

    # Used when loading a model
    def updateScalingVals(self, data):
        self.ScalingData = data

    # Encodes userdata by standardising and mapping categorical values
    def encode(self, UserData):
        try:
            for x, val in enumerate(UserData):
                # Converting into numerical data
                if val in self.CategoricalFeatureKeys.keys():
                    val = float(self.CategoricalFeatureKeys[val])
                else:
                    val = float(val)

                # Standardising the data
                UserData[x] = (val - self.ScalingData["means"][x]) / self.ScalingData["stds"][x]

            return UserData
        except Exception as ex:
            print(f"Encode Error: {ex}")