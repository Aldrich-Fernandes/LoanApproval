from NeuralNetwork import *
from Layer import *
from DataHandle import PreProcess
import tkinter as tk

class GUI:
    def __init__(self):    
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", exit)
        self.root.title("Home Loan Eligebility Application")
        self.root.geometry(f"800x600")

        self.Model, self.PreProcessor = setup()
        self.PredictFrame = tk.Frame(self.root, borderwidth=2)

        self.resultVal = tk.StringVar()
        self.resultVal.set("...")

    def LoadPredictionGUI(self):
        # generate Questionair 
        DataToGet = {'Applicant monthly income: ': -1,
             'Coapplicant monthly income: ': -1,
             'Loan amount (in thousands): ': -1,
             'Loan amount term (months): ': -1,
             'Credit history meet guildlines?: ': ["Yes", "No"],
             'Property area: ': ["Urban", "Semiurban", "Rural"]}
        self.UserData = [tk.StringVar() for x in range(len(DataToGet.keys()))]

        index = 0
        for key, data in DataToGet.items():
            label = tk.Label(self.PredictFrame, text=key)
            label.grid(row=index, column=0, padx=5, pady=5)

            if type(data) == list:
                for col, option in enumerate(data):
                    rb = tk.Radiobutton(self.PredictFrame, text=option, value=option, variable=self.UserData[index])
                    rb.grid(row=index, column=col+1, padx=5, pady=5)
            else:
                entry = tk.Entry(self.PredictFrame, textvariable=self.UserData[index]) # not appending to userdata
                entry.grid(row=index, column=1, padx=5, pady=5)

            index +=1

        self.ProcessBtn = tk.Button(self.PredictFrame, text="Enter", repeatinterval=5, command=self.ProcessUserData).grid(row=index, column=0, padx=5, pady=5)
        self.ResultLabel = tk.Label(self.PredictFrame, textvariable=self.resultVal).grid(row=index+1, column=0, padx=5, pady=5)

        self.PredictFrame.pack()

    def ProcessUserData(self):
        self.CollectedData = []
        print(self.UserData)
        for data in self.UserData:
            self.CollectedData.append(data.get())
        print(self.CollectedData)

        if not ('' in self.CollectedData or len(self.CollectedData) != 11):
            UserData = self.PreProcessor.encode(self.CollectedData)
            self.Model.Predict(UserData)

            result = self.Model.Result
            print(result)
            if round(result) == 1:
                txt = f"You a likely to be approved. Confidence = {result * 100}%"
            else:
                txt = f"You a unlikely to be approved. Confidence = {result * 100}%"
            print(txt)
            self.resultVal.set(txt)
        else:
            print(self.CollectedData)
            print("Missing or incorrect data.")

def SaveModel(Model, PreProcessor):
    FileName = input("Please enter a model name: ").lower()

    file = open(f"DataSet/Models/{FileName}.txt", "w")
    file.write(str(Model.Outputlayer.weights)+"\n")
    file.write(str(Model.Outputlayer.biases)+"\n")
    file.write(str(Model.Accuracy)+"\n")
    file.write(str(PreProcessor.ScalingData))
    file.close()

def LoadModel(FileName):
    try:
        PreProcessor = PreProcess()
        model = Model()

        file = open(f"DataSet/Models/{FileName}.txt", "r")
        model.Outputlayer.weights = eval(file.readline().rstrip())
        model.Outputlayer.biases = eval(file.readline().rstrip())
        model.Accuracy = eval(file.readline().rstrip())
        PreProcessor.ScalingData = eval(file.readline().rstrip())
        file.close()
        return model, PreProcessor

    except FileNotFoundError:
        print("File Doesnt exist. Returning to menu...")
        setup()

def setup():
    newModel = int(input("Menu: \n1). Load Default model \n2). Load Another Model \n3). Load new Model \nOption: "))
    if newModel == 1:
        File = "default"
        Model, PreProcessor = LoadModel(File)
    elif newModel == 2: 
        File = str(input("Enter Model name: "))
        Model, PreProcessor = LoadModel(File)
    elif newModel == 3:
        PreProcessor = PreProcess(New=True)
        TrainX, TrainY, TestX, TestY = PreProcessor.getData()

        Model = NeuralNetwork()
        Model.train(TrainX, TrainY, show=True)
        Model.test(TestX, TestY, showTests=True)
        save = int(input("Save model:\n 1). Yes\n 2). No \nOption: "))
        if save == 1:
            SaveModel(Model, PreProcessor)
    else:
        print("Incorrect option... Returning to menu.")
        setup()

    print("Here")
    return Model, PreProcessor

def main():
    myGUI = GUI()
    myGUI.LoadPredictionGUI()
    myGUI.root.mainloop()

#main()

def ModelTest():
    def getData():
        UserData = []
        DataToGet = {#'Gender: ': ["Male", "Female"],
                 #'Married: ': ["Yes", "No"],
                 #'Dependents (eg. number of childern/elderly): ': ["0", "1", "2", "+3"],
                 #'Education: ': ["Graduate", "Not Graduate"],
                 #'Self employed: ': ["Yes", "No"],
                 'Applicant monthly income: ': -1,
                 'Coapplicant monthly income: ': -1,
                 'Loan amount (in thousands): ': -1,
                 'Loan amount term (months): ': -1,
                 'Credit history meet guildlines?: ': ["Yes", "No"],
                 'Property area: ': ["Urban", "Semiurban", "Rural"]
                 }

        print("Please enter the following data.")
        for key, data in DataToGet.items():
            print("\n------------------------------------------------------\n",key)
            if type(data) == list:
                for x, val in enumerate(data):
                    print(f" {x+1}). {val}")
                choice = int(input("Choice: "))-1
                UserData.append(data[choice])
            else:
                UserData.append(int(input("Enter Data: ")))
        return UserData

    # Enter the name of the model to load (Press ENTER to train a new one): 

    PreProcessor = PreProcess(New=True)
    TrainX, TrainY, TestX, TestY = PreProcessor.getData()
    
    model = Model(Epochs=30)
    model.add(Layer(len(TrainX[0]), 1, "Sigmoid"))

    model.train(TrainX, TrainY, show=True)
    model.test(TestX, TestY, showTests=True)

    UserData = getData()
    UserData = PreProcessor.encode(UserData)
    model.Predict(UserData)
    print(model.Result)

ModelTest()