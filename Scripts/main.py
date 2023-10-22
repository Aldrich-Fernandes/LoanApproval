from NeuralNetwork import NeuralNetwork
from DataHandle import PreProcess, getData

def SaveModel():
    FileName = input("Please enter a model name: ")
    file = open(FileName, "w")
    file.write(str(model.Hiddenlayer.weights))
    file.write(str(model.Hiddenlayer.biases))
    file.write(str(model.Outputlayer.weights))
    file.write(str(model.Outputlayer.biases))
    file.close()


FileName = input("Press ENTER to train a new one: ")
# Enter the name of the model to load (Press ENTER to train a new one): 

PreProcessor = PreProcess(FileName)
TrainX, TrainY, TestX, TestY = PreProcessor.getData()

model = NeuralNetwork()
model.train(TrainX, TrainY, show=True)
model.graph()
model.test(TestX, TestY)

UserData = getData()
UserData = PreProcessor.encode(UserData)
model.Predict(UserData)

print(f"\n\nHidden layer:\n{model.Hiddenlayer.weights}\n\n {model.Hiddenlayer.biases}")
print(f"\n\nOutput layer:\n{model.Outputlayer.weights}\n\n {model.Outputlayer.biases}")
print(f"\n\n")


# Option to Train a new model -- Load a model - if empty train a new model
#           Load New -- Load a model
#                   --> Enter Data to predict approval