from NeuralNetwork import NeuralNetwork


Option = int(input("Do you want to train a new (1) or use a old (2) dataset:"))
if Option == 1:
    mode = "New"
    NoOfSamples = int(input("How many samples (20 - 600): "))
elif Option == 2:
    mode = "Load"
    NoOfSamples = 600

NeuralNetwork().train(mode, NoOfSamples)
# Option to Train a new model -- Load a model - if empty train a new model
#           Load New -- Load a model
#                   --> Enter Data to predict approval