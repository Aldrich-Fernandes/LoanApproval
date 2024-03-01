import os

def loadModel(filePath):
    with open(filePath,  "r") as file:
        scalingData = eval(file.readline().rstrip())
        weights = eval(file.readline().rstrip())
        biases = eval(file.readline().rstrip())
        print(f"ScalingData: {scalingData}")
        print(f"weights: {weights}")
        print(f"biases: {biases}")

print("C:\Python\Projects\PersonalProjects\LoanApproval\DataSet\Models\default.txt")
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
filename = os.path.join(parent_folder, "DataSet", "Models\default.txt")
loadModel(filename)
