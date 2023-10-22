from NeuralNetwork import NeuralNetwork
from DataHandle import PreProcess

def getData():
    UserData = []
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

def SaveModel():
    FileName = input("Please enter a model name: ")

    file = open(f"DataSet/Models/{FileName}.txt", "w")
    file.write(str(model.Hiddenlayer.weights)+"\n")
    file.write(str(model.Hiddenlayer.biases)+"\n")
    file.write(str(model.Outputlayer.weights)+"\n")
    file.write(str(model.Outputlayer.biases)+"\n")
    file.write(str(model.Accuracy)+"\n")
    file.write(str(PreProcessor.ScalingData))
    file.close()

def LoadModel():
    file = open(f"DataSet/Models/{FileName}.txt")
    model.Hiddenlayer.weights = eval(file.readline().rstrip())
    model.Hiddenlayer.biases = eval(file.readline().rstrip())
    model.Outputlayer.weights = eval(file.readline().rstrip())
    model.Outputlayer.biases = eval(file.readline().rstrip())
    model.Accuracy = eval(file.readline().rstrip())
    PreProcessor.ScalingData = eval(file.readline().rstrip())

FileName = input("Enter the name of the model to load (Press ENTER to train a new one): ")

if len(FileName) == 0:
    PreProcessor = PreProcess(New=True)
    TrainX, TrainY, TestX, TestY = PreProcessor.getData()

    model = NeuralNetwork()
    model.train(TrainX, TrainY, show=True)
    model.graph()
    model.test(TestX, TestY)

    CanSave = str(input("Save the model? (Y/N): ")).lower()
    if CanSave == "y":
        SaveModel()

else:
    model = NeuralNetwork()
    PreProcessor = PreProcess()
    LoadModel()
    print(f"Model Loaded Succesfully. (Acc={round(model.Accuracy*100, 2)})")

UserData = getData()
UserData = PreProcessor.encode(UserData)
model.Predict(UserData)

print("Thank you for using this service.")


# Option to Train a new model -- Load a model - if empty train a new model
#           Load New -- Load a model
#                   --> Enter Data to predict approval