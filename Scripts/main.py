from NeuralNetwork import LogisticRegression as Model
from DataHandle import PreProcess

import tkinter as tk
from tkinter import ttk, simpledialog

'''
User Interface

'''
class GUI:
    def __init__(self):
        self.__model = Model()                      # Neural Network model
        self.__model.addLayer(NoOfInputs=6, NoOfNeurons=1)    # adding Layers
        self.__PreProcessor = PreProcess()          # For preparing data

        # Loading the GUI window
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.__exit)
        self.root.title("Home Loan Eligibility Application")
        self.root.geometry("900x600")

        # Create a notebook (tabs container)
        self.__notebook = ttk.Notebook(self.root)
        self.__notebook.pack(expand=True, fill='both')

        # Setting variables
        self.__Font = ('Arial', 14)     # Default font to use
        self.__training = False         # Pauses user input if model is being trained

        # Adjustable hyperparameters
        self._updateValsTo = {"newEpoch": tk.IntVar(value = 25),
                             "newRegStr": tk.DoubleVar(value = 0.001),
                             "initialLr": tk.DoubleVar(value = 0.0001),
                             "decay": tk.DoubleVar(value = 0.00005),
                             "momentum": tk.DoubleVar(value = 0.95)}

        # Tkinter variable for outputs and user inputs
        self._resultVal = tk.StringVar(value="...")
        self._saveStatusVal = tk.StringVar(value="Default model loaded")
        self._fileName = tk.StringVar()

        self._CreateTabs()

    # Creates the different tabs (eg. Prediction and hyperparameter adjustment)
    def _CreateTabs(self):
        self._LoadMenu()

        self.__LoadPredictionGUI()
        self.__loadDefault()

        self.__LoadHyperparametersTab()


    # Dropdown menu that allows to change the model used (eg. train new or load default)
    def _LoadMenu(self):
        # Create a menu
        menuBar = tk.Menu(self.root)
        self.root.config(menu=menuBar)

        Menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="File", menu=Menu)
        Menu.add_command(label="Load Default Model", command=self.__loadDefault)
        Menu.add_command(label="Load other Model", command=self._loadModel)
        Menu.add_command(label="Generate New Model", command=self.__newModel)
        Menu.add_separator()
        Menu.add_command(label="Exit", command=self.__exit)

    # This tab allows users to change hyperparameters and retrain model
    def __LoadHyperparametersTab(self):
        # Create a new frame for the Hyperparameters tab
        hyperparameterFrame = ttk.Frame(self.__notebook)
        self.__notebook.add(hyperparameterFrame, text="Hyperparameters")

        # Add widgets for adjusting hyperparameters
        # Model Parameters
        tk.Label(hyperparameterFrame, text="Epochs:", font=self.__Font).grid(
            row=0, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["newEpoch"], font=self.__Font).grid(
            row=0, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Regularisation Strength:", font=self.__Font).grid(
            row=1, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["newRegStr"], font=self.__Font).grid(
            row=1, column=1, padx=10, pady=5)

        # Optimiser Parameters
        tk.Label(hyperparameterFrame, text="OPTIMISER PARAMETERS", font=self.__Font).grid(
            row=3, column=0, columnspan=2, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Initial Learning Rate:", font=self.__Font).grid(
            row=4, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["initialLr"], font=self.__Font).grid(
            row=4, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Decay:", font=self.__Font).grid(
            row=5, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["decay"], font=self.__Font).grid(
            row=5, column=1, padx=10, pady=5)

        tk.Label(hyperparameterFrame, text="Momentum:", font=self.__Font).grid(
            row=6, column=0, padx=10, pady=5)
        tk.Entry(hyperparameterFrame, textvariable=self._updateValsTo["momentum"], font=self.__Font).grid(
            row=6, column=1, padx=10, pady=5)

        # Retrain Button
        tk.Button(hyperparameterFrame, text="ReTrain Model", command=self.__updateHyperparameters).grid(
            row=8, column=0, columnspan=3, pady=10)

    # Updates the model's hyperparameters and retrains the model with the new hyperparameters
    def __updateHyperparameters(self):
        try:
            optimiserVals = [item.get() for item in list(self._updateValsTo.values())]
           
            # Apply hyperparameters to the model
            self.__model.updateEpoch(optimiserVals[0])
            self.__model.updateRegStr(optimiserVals[1])
            self.__model.configOptimiser(optimiserVals[2], optimiserVals[3], optimiserVals[4])

            print("Retraining model...")
            self.__newModel()

        except ValueError:
            print("Invalid input for epochs or regularisation strength. Please enter valid values.")

    # Main Prediction Interface
    def __LoadPredictionGUI(self):
        predictFrame = ttk.Frame(self.__notebook)
        self.__notebook.add(predictFrame, text="Prediction")

        # Data that is passed through the model
        DataToGet = {'Applicant monthly income: ': -1,
                     'Coapplicant monthly income: ': -1,
                     'Loan amount (in thousands): ': -1,
                     'Loan amount term (months): ': -1,
                     'Credit history meet guidelines?: ': ["Yes", "No"],
                     'Property area: ': ["Urban", "Semiurban", "Rural"]}
       
        # Creates a list of empty Tkinter string variables to user inputs
        self.UserData = [tk.StringVar() for _ in range(len(DataToGet.keys()))]
        for index, (key, data) in enumerate(DataToGet.items()):
            # Data Prompt
            tk.Label(predictFrame, text=key, font=self.__Font).grid(row=index, column=0, padx=5, pady=5)

            # User inputs
            if type(data) == list:  # Multiple choice option
                for col, option in enumerate(data):
                    tk.Radiobutton(predictFrame, text=option, value=option, variable=self.UserData[index],
                                   font=self.__Font).grid(row=index, column=col + 1, padx=5, pady=5)
            else:                   # Integer inputs
                tk.Entry(predictFrame, textvariable=self.UserData[index], font=self.__Font).grid(
                    row=index, column=1, padx=5, pady=5)

        # Button - Processes data
        tk.Button(predictFrame, text="Enter", font=self.__Font, command=self.__ProcessUserData).grid(
            row=len(DataToGet), column=0, columnspan=2, pady=10)

        # Updateable status prompts
        self._ResultLabel = tk.Label(predictFrame, textvariable=self._resultVal, font=self.__Font)
        self._ResultLabel.grid(row=len(DataToGet) + 1, column=1, columnspan=2, pady=10)
        self._saveStatusLabel = tk.Label(predictFrame, textvariable=self._saveStatusVal, font=self.__Font)
        self._saveStatusLabel.grid(row=len(DataToGet)+4, columnspan=4, pady=10, sticky="WE")

        tk.Button(predictFrame, text="Save Model", font=self.__Font, command=self._saveModel).grid(
            row=len(DataToGet)+3, column=0, columnspan=2, pady=10)
        tk.Entry(predictFrame, textvariable=self._fileName, font=self.__Font).grid(
            row=len(DataToGet)+3, column=2,columnspan=2, pady=10)

    # Processes UserData through the model
    def __ProcessUserData(self):
        if not self.__training:
            try:
                # Converts Tkinter variables into normal data to be preprocessed
                self.CollectedData = []
                for data in self.UserData:
                    self.CollectedData.append(data.get())

                # Ensures all data is valid
                if not ('' in self.CollectedData or len(self.CollectedData) != 6):
                    # Encodes data - More info in DataHandle.py file
                    UserData = self.__PreProcessor.encode(self.CollectedData)
                    self.__model.Predict([UserData])
                    result = round(self.__model.Result * 100)
                   
                    status = f"You have a {result}% chance of being Approved"
                else:
                    status = "Missing or incorrect data."
            except ValueError:
                status = "Error in userdata"
        else:
            status = "Model is training. Please wait."

        # Updates result prompts
        self._resultVal.set(status)
        self._ResultLabel.config(textvariable=self._resultVal)

    # Generates a new Neural network model using default hyperparameters
    def __newModel(self):
        self._saveStatusVal.set("Generating new model...")
        self._saveStatusLabel.config(textvariable=self._saveStatusVal)

        # Restarts already initialised Preprocess object
        self.__PreProcessor.newDataset()
        TrainX, TrainY, TestX, TestY = self.__PreProcessor.getData()

        valid = False
        while not valid:
            # Trains model with new random data
            self.__model.train(TrainX, TrainY)
            self.__model.test(TestX, TestY)
            accuracy = self.__model.Accuracy

            # Due to model limitations the model will be trained again if it is not at an acceptable accuracy.
            # This is to ensure that the model doesn't overfit ('memorise' training data) or converge on one
            # output (eg. always output 1 prediction like 0.643 or 64.3%)
            if accuracy > 0.74:
                status = f"Valid model generated - Accuracy: {accuracy}    |    Unsaved"
                self.__training, valid = False, True

                self._saveStatusVal.set(status)
                self._saveStatusLabel.config(textvariable=self._saveStatusVal)
            else:
                self.__model.resetLayers()
                self.__PreProcessor.newDataset()
                TrainX, TrainY, TestX, TestY = self.__PreProcessor.getData()
       
    # Saves model data so that it can be loaded later
    def _saveModel(self):
        if self._fileName.get() != "":
            filePath = f"DataSet\\Models\\{self._fileName.get()}.txt"
            status = self.__model.saveModel(filePath, self.__PreProcessor.getScalingData())
            self._saveStatusVal.set(status)
            self._saveStatusLabel.config(textvariable=self._saveStatusVal)

    # Loads saved models
    def _loadModel(self):
        modelName = simpledialog.askstring("Load Another Model", "Enter model name:")
        filePath = f"DataSet\\Models\\{modelName}.txt"
        scalingData = self.__model.loadModel(filePath)

        if scalingData == None:
            print("File not found. Loading default...")
            self.__loadDefault()
        else:
            self.__PreProcessor.setScalingData(scalingData)

    # Loads the a pretrained model to save time when launching the program
    def __loadDefault(self):
        filePath = "DataSet\\Models\\default.txt"
        self.__model = Model()      # Resets model incase default hyperparameters were changed.
        self.__model.addLayer(NoOfInputs=6, NoOfNeurons=1)
        scalingData = self.__model.loadModel(filePath)

        self.__PreProcessor.setScalingData(scalingData)
        status = "Default model loaded"
        self._saveStatusVal.set(status)
        self._saveStatusLabel.config(textvariable=self._saveStatusVal)

    # Terminates the window
    def __exit(self):
        self.root.destroy()

# Main loop
def main():
    myGUI = GUI()
    myGUI.root.mainloop()
main()
