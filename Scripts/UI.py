import tkinter as tk

class GUI:
    def __init__(self):    
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", exit)
        self.root.title("Home Loan Approval Predictor")
        self.root.geometry(f"700x500")

        self.CollectedData = []

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

        ProcessBtn = tk.Button(self.PredictFrame, text="Enter", repeatinterval=5, command=self.UpdateUserData).grid(row=index, column=0, padx=5, pady=5)


        self.PredictFrame.pack()

    def showResult(self, result):
        if round(result) == 1:
            txt = f"You a likely to be approved. Confidence = {result * 100}%"
        else:
            txt = f"You a unlikely to be approved. Confidence = {result * 100}%"

        #print("Here")
        #tk.Label(self.PredictFrame, text=f"Model Loaded Succesfully. (Acc={round(accuracy*100, 2)}%)", font=('Arial', 20)).grid(row=14, column=0)
        #tk.Label(self.PredictFrame, text=txt).grid(row=15, column=0)

    def UpdateUserData(self):
        self.CollectedData = []
        for data in self.UserData:
            self.CollectedData.append(data.get())
        return self.CollectedData