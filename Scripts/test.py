from DataHandle import DataMethod as DM
def DotProduct(arr1, arr2): # change from vector dot product to matrix dot product
    if not isinstance(arr1, list): 
        arr1 = [[float(arr1) for i in range(len(arr2))] for j in range(len(arr2[0]))] # if arr2 = (2,3) creates arr = (3, 2)
    elif not isinstance(arr2, list):
        arr2 = [[float(arr2) for i in range(len(arr1))] for j in range(len(arr1[0]))]

    arr2Shape = [len(arr2), len(arr2[0])]
    arr1Shape = [len(arr1), len(arr1[0])]
    if arr1Shape[1] == arr2Shape[0]: # valid matrixes to multiply
        Output = []
        for rowIndex, row in enumerate(arr1):
            Output.append([])
            for column in DM.Transpose(arr2):
                Output[rowIndex].append(sum(a*b for a,b in zip(row, column)))
        print(Output)
    else:
        print(f"Not capable of dotting as \n  Array1:{arr1Shape}\n  Array2:{arr2Shape}")
        input()

def optimizer():
    import matplotlib.pyplot as plt

    # Define the number of epochs and initial learning rate
    num_epochs = 100
    initial_learning_rate = 0.01

    # Create a learning rate schedule with linear decay
    def learning_rate_schedule(epoch):
        return max(InitialLearningRate / (1 + decay * iter),
                                           minimumLearningRate)

    # Lists to store the learning rates and epochs
    learning_rates = []
    epochs = list(range(num_epochs))

    # Generate learning rates for each epoch
    for epoch in range(num_epochs):
        current_learning_rate = learning_rate_schedule(epoch)
        learning_rates.append(current_learning_rate)

    # Plot the learning rate curve
    plt.plot(epochs, learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.show()

optimizer()