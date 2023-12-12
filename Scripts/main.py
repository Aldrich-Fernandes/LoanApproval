from NeuralNetwork import *
from Layer import *
from DataHandle import PreProcess
import tkinter as tk

class GUI:
    def __init__(self):
        self.__model, self.__PreProcessor = self.setup()
        self.__Font = ('Arial', 14)
        
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.exit)
        self.root.title("Home Loan Eligibility Application")
        self.root.geometry("800x600")

        self.__PredictFrame = tk.Frame(self.root, borderwidth=2)
        self.__PredictFrame.pack(padx=20, pady=20)

        self.__resultVal = tk.StringVar()
        self.__resultVal.set("...")

    def exit(self):
        self.root.destroy()

    def setup(self):
        loop = True
        while loop:
            PreProcessor = PreProcess(New=True)
            TrainX, TrainY, TestX, TestY = PreProcessor.getData()
            
            model = Model(Epochs=15)
            model.add(Layer(len(TrainX[0]), 1, "Sigmoid"))

            model.train(TrainX, TrainY)
            model.test(TestX, TestY)
            accuracy = model.Accuracy
    
            if accuracy <= 0.7:
                print(f"Unstable Model. Retraining...\n")
            else:
                return model, PreProcessor

    def LoadPredictionGUI(self):
        DataToGet = {'Applicant monthly income: ': -1,
                     'Coapplicant monthly income: ': -1,
                     'Loan amount (in thousands): ': -1,
                     'Loan amount term (months): ': -1,
                     'Credit history meet guidelines?: ': ["Yes", "No"],
                     'Property area: ': ["Urban", "Semiurban", "Rural"]}
        self.UserData = [tk.StringVar() for _ in range(len(DataToGet.keys()))]

        for index, (key, data) in enumerate(DataToGet.items()):
            label = tk.Label(self.__PredictFrame, text=key, font=self.__Font)
            label.grid(row=index, column=0, sticky="e", padx=5, pady=5)

            if type(data) == list:
                for col, option in enumerate(data):
                    rb = tk.Radiobutton(self.__PredictFrame, text=option, value=option, variable=self.UserData[index], font=self.__Font)
                    rb.grid(row=index, column=col + 1, padx=5, pady=5)
            else:
                entry = tk.Entry(self.__PredictFrame, textvariable=self.UserData[index], font=self.__Font)
                entry.grid(row=index, column=1, padx=5, pady=5)

        self.ProcessBtn = tk.Button(self.__PredictFrame, text="Enter", font=self.__Font, command=self.ProcessUserData)
        self.ProcessBtn.grid(row=len(DataToGet), column=0, columnspan=2, pady=10)

        self.ResultLabel = tk.Label(self.__PredictFrame, textvariable=self.__resultVal, font=self.__Font)
        self.ResultLabel.grid(row=len(DataToGet) + 1, column=0, columnspan=2, pady=10)

    def ProcessUserData(self):
        self.CollectedData = []
        for data in self.UserData:
            self.CollectedData.append(data.get())

        if not ('' in self.CollectedData or len(self.CollectedData) != 6):
            UserData = self.__PreProcessor.encode(self.CollectedData)
            self.__model.Predict([UserData])

            result = self.__model.Result
            
            self.__resultVal.set(f"You have a {result*100}% chance of being Approved")
            self.ResultLabel.config(textvariable=self.__resultVal)
        else:
            print(self.CollectedData)
            print("Missing or incorrect data.")

def main():
    myGUI = GUI()
    myGUI.LoadPredictionGUI()
    myGUI.root.mainloop()

main()

def ModelTest():
    def getData():
        UserData = []
        DataToGet = {
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

    loop = True
    while loop:
        PreProcessor = PreProcess(New=True)
        TrainX, TrainY, TestX, TestY = PreProcessor.getData()
        
        model = Model(Epochs=15)
        model.add(Layer(len(TrainX[0]), 1, "Sigmoid"))

        model.train(TrainX, TrainY)
        model.test(TestX, TestY)
        accuracy = model.Accuracy

        if accuracy <= 0.7:
            print(f"Unstable Model. Retraining...\n")
        else:
            loop = False

    UserData = getData()
    UserData = PreProcessor.encode(UserData)
    model.Predict(UserData)
    print(model.Result)

#ModelTest()