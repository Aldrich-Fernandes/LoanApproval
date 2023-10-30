from NeuralNetwork import NeuralNetwork
from DataHandle import PreProcess
import tkinter as tk

class GUI:
    def __init__(self):    
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", exit)
        self.root.title("Home Loan Approval Predictor")
        self.root.geometry(f"700x500")

        self.Model, self.PreProcessor = setup()

    def LoadPredictionGUI(self):
        self.PredictFrame = tk.Frame(self.root, borderwidth=2)

        # generate Questionair 
        DataToGet = {'Gender: ': ["Male", "Female"],
             'Married: ': ["Yes", "No"],
             'Dependents (eg. number of childern/elderly): ': ["0", "1", "2", "+3"],
             'Education: ': ["Graduate", "Not Graduate"],
             'Self employed: ': ["Yes", "No"],
             'Applicant monthly income: ': -1,
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
        
        self.PredictFrame.pack()

    def ProcessUserData(self):
        self.CollectedData = []
        for data in self.UserData:
            self.CollectedData.append(data.get())
        
        if not ('' in self.CollectedData or len(self.CollectedData) != 11):
            UserData = self.PreProcessor.encode(self.CollectedData)
            self.Model.Predict(UserData)

            result = self.Model.Result
            if round(result) == 1:
                txt = f"You a likely to be approved. Confidence = {result * 100}%"
            else:
                txt = f"You a unlikely to be approved. Confidence = {result * 100}%"

            self.ResultLabel = tk.Label(self.PredictFrame, text=txt).grid(row=13, column=0, padx=5, pady=5)
        else:
            print(self.CollectedData)
            print("Missing or incorrect data.")

def SaveModel(Model, PreProcessor):
    FileName = input("Please enter a model name: ").lower()

    file = open(f"DataSet/Models/{FileName}.txt", "w")
    file.write(str(Model.Hiddenlayer.weights)+"\n")
    file.write(str(Model.Hiddenlayer.biases)+"\n")
    file.write(str(Model.Outputlayer.weights)+"\n")
    file.write(str(Model.Outputlayer.biases)+"\n")
    file.write(str(Model.Accuracy)+"\n")
    file.write(str(PreProcessor.ScalingData))
    file.close()

def LoadModel(FileName):
    try:
        PreProcessor = PreProcess()
        Model = NeuralNetwork()

        file = open(f"DataSet/Models/{FileName}.txt", "r")
        Model.Hiddenlayer.weights = eval(file.readline().rstrip())
        Model.Hiddenlayer.biases = eval(file.readline().rstrip())
        Model.Outputlayer.weights = eval(file.readline().rstrip())
        Model.Outputlayer.biases = eval(file.readline().rstrip())
        Model.Accuracy = eval(file.readline().rstrip())
        PreProcessor.ScalingData = eval(file.readline().rstrip())
        file.close()
        return Model, PreProcessor
    
    except :
        print("File Doesnt exist. Returning to menu...")
        main()

def setup():
    newModel = int(input("Menu: \n1). Load Default model \n2). Load Another Model \n3). Load new Model"))
    if newModel == 1:
        File = "default"
        Model, PreProcessor = LoadModel(FileName=File)
    elif newModel == 2: 
        File = str(input("Enter Model name: "))
        Model, PreProcessor = LoadModel(FileName=File)
    elif newModel == 3:
        Model = NeuralNetwork()
        PreProcessor = PreProcess(New=True)
        save = int(input("Save model:\n 1). Yes\n 2). No \nOption: "))
        if save == 1:
            SaveModel(Model, PreProcessor)
    else:
        print("Incorrect option... Returning to menu.")
        main()
    return Model, PreProcessor

def main():
    myGUI = GUI()
    myGUI.LoadPredictionGUI()
    myGUI.root.mainloop()

main()