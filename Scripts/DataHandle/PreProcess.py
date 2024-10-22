from .DataUtils import DataMethod

import random

'''
Raw Data cannot be processed by the model. Therefore both training and userdata have to be adjusted prior to 
training or prediction

'''
class Preprocess:
    def __init__(self, path=r"DataSet//HomeLoanTrain.csv"):
        # Initial data holders for training data
        self._TrainX = []
        self._TrainY = []


        # Data specific to the training data used
        self.__featuresToRemove = ["Loan_ID", "Self_Employed", "Gender", "Education", "Married", "Dependents"]
        self.__CategoricalFeatureKeys = {"Y": 1., "Yes": 1., "Male": 1., "Graduate": 1., "Urban": 1., 
                                    "N": 0., "No": 0., "Female": 0., "Not Graduate": 0., "Semiurban": 0.,
                                    "Rural": 2., "3+": 2.}
        self._ScalingData = {'means': [], 'stds': []}
        self.__path = path

    # Generates a new random dataset
    def newDataset(self):
        # Extract data from file and selecting valid entries/features
        Dataset = DataMethod.CsvToArray(self.__path)
        Dataset = self.__RemoveFeatures(Dataset)
        Dataset = self.__AdjustSkew(Dataset) 

        # Feature Engineering - Making data usable for the model

        self.__FeatureColumns = DataMethod.Transpose(Dataset)

        self.__ConvertToInteger()

        self._TrainY = self.__FeatureColumns.pop()

        self.__Standardisation()

        self._TrainX = DataMethod.Transpose(self.__FeatureColumns)

        self._TrainX, self._TrainY = DataMethod.ShuffleData(self._TrainX, self._TrainY)

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
        for ColInd, features in enumerate(self.__FeatureColumns):
            for EleInd, element in enumerate(features):
                try:                        # For data that is already numerical
                    self.__FeatureColumns[ColInd][EleInd] = float(element)
                except ValueError:          # For data that is categorical
                    if element not in self.__CategoricalFeatureKeys.keys():
                        self.__CategoricalFeatureKeys[str(element)] = sum([ord(x) for x in element]) / 16
                    self.__FeatureColumns[ColInd][EleInd] = self.__CategoricalFeatureKeys[str(element)]

    # Z-score normalisation formula: (data - mean) / standard deviation
    # Improves interpretability and model performance by removing scale difference between features 
    def __Standardisation(self):
        for ind, feature in enumerate(self.__FeatureColumns):
            mean = sum(feature) / len(feature)
            StandardDeviation = ((sum([x**2 for x in feature]) / len(feature)) - mean**2) ** 0.5

            # To use when scale userdata properly for the dataset
            self._ScalingData['means'].append(mean)
            self._ScalingData['stds'].append(StandardDeviation)

            # Applying the standardisation
            try:
                self.__FeatureColumns[ind] = [(i - mean) / StandardDeviation for i in feature]
            except ZeroDivisionError as EXP:
                print(f"Mean: {mean} \nSTD: {StandardDeviation}")
                print(f"FeatureColumn: {self.__FeatureColumns[ind]} \n {EXP}")

    # Splits the dataset into training and test sets and returns the data
    def getData(self, split=0.2): # default: 80-20 split
        NumOfTrainData = round(len(self._TrainX) * split)
        TestX = [self._TrainX.pop() for _ in range(NumOfTrainData)]
        TestY = [self._TrainY.pop() for _ in range(NumOfTrainData)]
        return self._TrainX, self._TrainY, TestX, TestY

    # Getters and Setters used when saving and loading a model
    def setScalingData(self, data):
        self._ScalingData = data
    
    def getScalingData(self):
        return self._ScalingData

    # Encodes userdata by standardising and mapping categorical values
    def encode(self, UserData):
        try:
            for x, val in enumerate(UserData):
                # Converting into numerical data
                if val in self.__CategoricalFeatureKeys.keys():
                    val = float(self.__CategoricalFeatureKeys[val])
                else:
                    val = float(val)

                # Standardising the data
                UserData[x] = (val - self._ScalingData["means"][x]) / self._ScalingData["stds"][x]

            return UserData
        except Exception as ex:
            print(f"Encode Error: {ex}")