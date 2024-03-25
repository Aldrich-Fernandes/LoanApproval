from NeuralNetwork import LogisticRegression as Model
from DataHandle import Preprocess

model = Model()                      # Neural Network model
model.addLayer(NoOfInputs=6, NoOfNeurons=1)    # adding Layers
Preprocessor = Preprocess()          # For preparing data

Preprocessor.newDataset()
TrainX, TrainY, TestX, TestY = Preprocessor.getData()

model.train(TrainX, TrainY, canGraph=True)