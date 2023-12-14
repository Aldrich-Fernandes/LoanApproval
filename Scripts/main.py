from NeuralNetwork import *
from Layer import *
from DataHandle import PreProcess

import tkinter as tk
from tkinter import ttk

class GUI:
    def __init__(self):
        self.__Font = ('Arial', 14)
        
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.__exit)
        self.root.title("Home Loan Eligibility Application")
        self.root.geometry("800x600")

        # Create a notebook (tabs container)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both')

        # Create tabs
        self.LoadPredictionGUI()
        self.LoadHyperparametersTab()

        # save a old model and by default load that data so that a new model doesnt need to be trained. They can retrain a model using the hyperparameter tab.
        # If they have pressed the retrain button, then a save model button will pop up which wil allow so. 
        # they can also choose to load a model in a dropdown menu which aslo has a exit button.

        self.__model, self.__PreProcessor = self.__setup()

    def LoadHyperparametersTab(self):
        # Create a new frame for the Hyperparameters tab
        hyperparameterFrame = ttk.Frame(self.notebook)
        self.notebook.add(hyperparameterFrame, text="Hyperparameters")

        self.__updateValsTo = {"newEpoch": tk.IntVar(value = 25),
                             "newRegStr": tk.DoubleVar(value = 0.001),
                             "initialLearningRate": tk.DoubleVar(value = 0.0001),
                             "decay": tk.DoubleVar(value = 0.00005),
                             "momentum": tk.DoubleVar(value=0.95),
                             "mode": tk.StringVar(value= "Linear")}

        # Add widgets for adjusting hyperparameters

        # Model Parameters
        tk.Label(hyperparameterFrame, text="Epochs:", font=self.__Font).grid(row=0, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self.__updateValsTo["newEpoch"], font=self.__Font).grid(row=0, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Regularization Strength:", font=self.__Font).grid(row=1, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self.__updateValsTo["newRegStr"], font=self.__Font).grid(row=1, column=1, padx=10, pady=5)

        # Optimiser Parameters
        tk.Label(hyperparameterFrame, text="OPTIMISER PARAMETERS", font=self.__Font).grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Initial Learning Rate:", font=self.__Font).grid(row=4, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self.__updateValsTo["initialLearningRate"], font=self.__Font).grid(row=4, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Decay:", font=self.__Font).grid(row=5, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self.__updateValsTo["decay"], font=self.__Font).grid(row=5, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Momentum:", font=self.__Font).grid(row=6, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self.__updateValsTo["momentum"], font=self.__Font).grid(row=6, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Mode:", font=self.__Font).grid(row=7, column=0, padx=10, pady=5)
        tk.Radiobutton(hyperparameterFrame, text="Linear", value="Linear", variable=self.__updateValsTo["momentum"], font=self.__Font).grid(row=7, column=1, padx=10, pady=5)
        tk.Radiobutton(hyperparameterFrame, text="Exponential", value="Exponential", variable=self.__updateValsTo["momentum"], font=self.__Font).grid(row=7, column=2, padx=10, pady=5)
        

        tk.Button(hyperparameterFrame, text="ReTrain Model", command=self.updateHyperparameters).grid(row=8, column=0, columnspan=2, pady=10)

    def updateHyperparameters(self):
        try:

            optimiserVals = [eval(item) for item in self.__updateValsTo.items()][2:]
            
            # Apply hyperparameters to the model
            self.__model.updateEpoch(eval(self.__updateValsTo["newEpoch"].get()))
            self.__model.updateRegStr(eval(self.__updateValsTo["newRegStr"].get()))

            self.__model.configOptimizer(optimiserVals[0], optimiserVals[1], optimiserVals[2], optimiserVals[3])

            print("Retraining model...")
            self.__restartModel()
        
        except ValueError:
            print("Invalid input for epochs or regularization strength. Please enter valid values.")

    def __exit(self):
        self.root.destroy()

    def __setup(self):
        loop = True
        while loop:
            PreProcessor = PreProcess(New=True)
            TrainX, TrainY, TestX, TestY = PreProcessor.getData()
            
            model = Model()
            model.add(Layer(len(TrainX[0]), 1, "Sigmoid"))

            model.train(TrainX, TrainY)
            model.test(TestX, TestY)
            accuracy = model.Accuracy
    
            if accuracy <= 0.7:
                print(f"Unstable Model. Retraining...\n")
            else:
                return model, PreProcessor
            
    def __restartModel(self):
        for x in range(8):
            PreProcessor = PreProcess(New=True)
            TrainX, TrainY, TestX, TestY = PreProcessor.getData()

            self.__model.train(TrainX, TrainY)
            self.__model.test(TestX, TestY)
            accuracy = self.__model.Accuracy
    
            if accuracy <= 0.7:
                print(f"Unstable Model. Retraining...\n")

        print("Too many attempts... Aborting Application.")
        self.__exit()

    def LoadPredictionGUI(self):
        predictFrame = ttk.Frame(self.notebook)
        self.notebook.add(predictFrame, text="Prediction")

        self.__resultVal = tk.StringVar()
        self.__resultVal.set("...")

        DataToGet = {'Applicant monthly income: ': -1,
                     'Coapplicant monthly income: ': -1,
                     'Loan amount (in thousands): ': -1,
                     'Loan amount term (months): ': -1,
                     'Credit history meet guidelines?: ': ["Yes", "No"],
                     'Property area: ': ["Urban", "Semiurban", "Rural"]}
        self.UserData = [tk.StringVar() for _ in range(len(DataToGet.keys()))]

        for index, (key, data) in enumerate(DataToGet.items()):
            tk.Label(predictFrame, text=key, font=self.__Font).grid(row=index, column=0, padx=5, pady=5)

            if type(data) == list:
                for col, option in enumerate(data):
                    tk.Radiobutton(predictFrame, text=option, value=option, variable=self.UserData[index], font=self.__Font).grid(row=index, column=col + 1, padx=5, pady=5)
            else:
                tk.Entry(predictFrame, textvariable=self.UserData[index], font=self.__Font).grid(row=index, column=1, padx=5, pady=5)

        tk.Button(predictFrame, text="Enter", font=self.__Font, command=self.__ProcessUserData).grid(row=len(DataToGet), column=0, columnspan=2, pady=10)

        self.ResultLabel = tk.Label(predictFrame, textvariable=self.__resultVal, font=self.__Font)
        self.ResultLabel.grid(row=len(DataToGet) + 1, column=0, columnspan=2, pady=10)

    def __ProcessUserData(self):
        self.CollectedData = []
        for data in self.UserData:
            self.CollectedData.append(data.get())

        if not ('' in self.CollectedData or len(self.CollectedData) != 6):
            UserData = self.__PreProcessor.encode(self.CollectedData)
            self.__model.Predict([UserData])

            result = round(self.__model.Result, 4)
            
            self.__resultVal.set(f"You have a {result*100}% chance of being Approved")
            self.ResultLabel.config(textvariable=self.__resultVal)
        else:
            print(self.CollectedData)
            print("Missing or incorrect data.")

def main():
    myGUI = GUI()
    myGUI.root.mainloop()

main()

def test():
    PreProcessor = PreProcess(New=True)
    TrainX, TrainY, TestX, TestY = PreProcessor.getData()

    model = Model(Epochs=20)
    model.add(Layer(len(TrainX[0]), 1, "Sigmoid"))

    model.train(TrainX, TrainY, canGraph=True)
    model.test(TestX, TestY, showTests=True)

#test()
