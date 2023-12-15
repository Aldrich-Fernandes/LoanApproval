from NeuralNetwork import *
from Layer import *
from DataHandle import PreProcess

import tkinter as tk
from tkinter import ttk, simpledialog

class GUI:
    def __init__(self):
        self.__model = Model()
        self.__model.add(Layer(6, 1, "Sigmoid"))
        self.__PreProcessor = PreProcess()

        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.__exit)
        self.root.title("Home Loan Eligibility Application")
        self.root.geometry("900x600")

        self.__Font = ('Arial', 14)
        self.__training = False

        self._resultVal = tk.StringVar(value="...")
        self._saveStatusVal = tk.StringVar(value="Unsaved")
        self._fileName = tk.StringVar()

        self._CreateTabs()

        self.__loadDefault()

    def _LoadMenu(self):
        # Create a menu
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Default Model", command=self.__loadDefault)
        file_menu.add_command(label="Load other Model", command=self._loadModel)
        file_menu.add_command(label="Generate New Model", command=self.__newModel)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.__exit)

    def _CreateTabs(self):
        # Create a notebook (tabs container)
        self.__notebook = ttk.Notebook(self.root)
        self.__notebook.pack(expand=True, fill='both')

        # Create tabs
        self._LoadMenu()
        self._LoadPredictionGUI()
        self._LoadHyperparametersTab()

    def _LoadHyperparametersTab(self):
        # Create a new frame for the Hyperparameters tab
        hyperparameterFrame = ttk.Frame(self.__notebook)
        self.__notebook.add(hyperparameterFrame, text="Hyperparameters")

        self._updateValsTo = {"newEpoch": tk.IntVar(value = 25),
                             "newRegStr": tk.DoubleVar(value = 0.001),
                             "initialLearningRate": tk.DoubleVar(value = 0.0001),
                             "decay": tk.DoubleVar(value = 0.00005),
                             "momentum": tk.DoubleVar(value = 0.95)}

        # Add widgets for adjusting hyperparameters

        # Model Parameters
        tk.Label(hyperparameterFrame, text="Epochs:", font=self.__Font).grid(row=0, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["newEpoch"], font=self.__Font).grid(row=0, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Regularization Strength:", font=self.__Font).grid(row=1, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["newRegStr"], font=self.__Font).grid(row=1, column=1, padx=10, pady=5)

        # Optimiser Parameters
        tk.Label(hyperparameterFrame, text="OPTIMISER PARAMETERS", font=self.__Font).grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Initial Learning Rate:", font=self.__Font).grid(row=4, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["initialLearningRate"], font=self.__Font).grid(row=4, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Decay:", font=self.__Font).grid(row=5, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["decay"], font=self.__Font).grid(row=5, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Momentum:", font=self.__Font).grid(row=6, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["momentum"], font=self.__Font).grid(row=6, column=1, padx=10, pady=5)

        tk.Button(hyperparameterFrame, text="ReTrain Model", command=self.__updateHyperparameters).grid(row=8, column=0, columnspan=3, pady=10)

    def __updateHyperparameters(self):
        try:

            optimiserVals = [item.get() for item in list(self._updateValsTo.values())]
            
            # Apply hyperparameters to the model
            self.__model.updateEpoch(optimiserVals[0])
            self.__model.updateRegStr(optimiserVals[1])

            self.__model.configOptimizer(optimiserVals[2], optimiserVals[3], optimiserVals[4])

            print("Retraining model...")
            
            self.__newModel()

        except ValueError:
            print("Invalid input for epochs or regularization strength. Please enter valid values.")

    def _LoadPredictionGUI(self):
        predictFrame = ttk.Frame(self.__notebook)
        self.__notebook.add(predictFrame, text="Prediction")

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

        self._ResultLabel = tk.Label(predictFrame, textvariable=self._resultVal, font=self.__Font)
        self._ResultLabel.grid(row=len(DataToGet) + 1, column=0, columnspan=2, pady=10)

        self._saveStatusLabel = tk.Label(predictFrame, textvariable=self._saveStatusVal, font=self.__Font)
        self._saveStatusLabel.grid(row=len(DataToGet)+4, column=0, padx=5, pady=5)

        tk.Button(predictFrame, text="Save Model", font=self.__Font, command=self._saveModel).grid(row=len(DataToGet)+3, column=0, columnspan=2, pady=10)
        tk.Entry(predictFrame, textvariable=self._fileName, font=self.__Font).grid(row=len(DataToGet)+3, column=2, padx=5, pady=10)

    def __ProcessUserData(self):
        if not self.__training:
            try:
                self.CollectedData = []
                for data in self.UserData:
                    self.CollectedData.append(data.get())

                if not ('' in self.CollectedData or len(self.CollectedData) != 6):
                    UserData = self.__PreProcessor.encode(self.CollectedData)
                    self.__model.Predict([UserData])

                    result = round(self.__model.Result, 4)
                    
                    status = f"You have a {result*100}% chance of being Approved"
                else:
                    status = "Missing or incorrect data."
            except ValueError:
                status = "Error in userdata"
        else:
            status = "Model is training. Please wait."

        self._resultVal.set(status)
        self._ResultLabel.config(textvariable=self._resultVal)

    def __newModel(self):
        self.__PreProcessor.newDataset()
        TrainX, TrainY, TestX, TestY = self.__PreProcessor.getData()

        while True:

            self.__model.train(TrainX, TrainY)
            self.__model.test(TestX, TestY)
            accuracy = self.__model.Accuracy
    
            if accuracy <= 0.7:
                print(f"Unstable Model - Accuracy: {accuracy}. \nRetraining...\n")
                self.__model.resetLayers()
                self.__PreProcessor.newDataset()
                TrainX, TrainY, TestX, TestY = self.__PreProcessor.getData()
            else:
                print(f"Model Successful - Accuracy: {accuracy}")
                self.__training = False
                break

    def _saveModel(self):
        if self._fileName.get() != "":
            filePath = f"DataSet\\Models\\{self._fileName.get()}.txt"
            status = self.__model.saveModel(filePath, self.__PreProcessor.ScalingData)
            self._saveStatusVal.set(status)
            self._saveStatusLabel.config(textvariable=self._saveStatusVal)

    def _loadModel(self):
        modelName = simpledialog.askstring("Load Another Model", "Enter model name:")
        filePath = f"DataSet\\Models\\{modelName}.txt"
        scalingData = self.__model.loadModel(filePath)
        self.__PreProcessor.updateScalingVals(scalingData)

    def __loadDefault(self):
        filePath = f"DataSet\\Models\\default.txt"
        scalingData = self.__model.loadModel(filePath)
        self.__PreProcessor.updateScalingVals(scalingData)

    def __exit(self):
        self.root.destroy()

def main():
    myGUI = GUI()
    myGUI.root.mainloop()

main()
