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