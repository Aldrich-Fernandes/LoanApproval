import csv
import random

'''
Commonly used maths functions processes by other programs

'''
class DataMethod:
    @staticmethod
    def CsvToArray(path): 
        # loads all data ignoring entries with missing data
        table = []
        with open(path, "r") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                if '' not in row:
                    table.append(row)
        file.close()

        return table

    # Swaps row and columns (eg. [[2, 4, 3], [5, 6, 7]] --> [[2, 5], [4, 6], [3, 7]] )
    @staticmethod
    def Transpose(array):
        return [[array[x][y] for x in range(len(array))] for y in range(len(array[0]))]

    # Performs Dot product on two valid matrices
    # valid if b = c for shapes (a, b) (c, d) | (Rows, Columns)
    @staticmethod
    def DotProduct(arr1, arr2):
        # Ensures they are matrices
        if isinstance(arr1[0], list) and isinstance(arr2[0], list):
            # Checks if they are valid 
            if len(arr1[0]) != len(arr2): 
                arr1Shape = (len(arr1),len(arr1[0]))
                arr2Shape = (len(arr2),len(arr2[0]))

                raise ValueError(f"arr1: ({arr1Shape}) and arr2: ({arr2Shape}) are not valid.")
            
            # Empty matrix to hold results
            result = [[0 for _ in range(len(arr2[0]))] for _ in range(len(arr1))]

            for i, row in enumerate(arr1):                             # For each row in arr1 
                for j, column in enumerate(DataMethod.Transpose(arr2)):# For each column in arr2
                    result[i][j] = sum(DataMethod.Multiply(row, column))
    
            return result
        else:
            raise ValueError("Parameters aren't matrices")
    
    # Performs multiplications involving arrays 
    @staticmethod
    def Multiply(arr1, arr2): # Limitation: dimensions of arr1 Must be <= dimensions of arr2
        try:
            if not isinstance(arr1, list): # Ensures it is at least 1 dimensional
                if isinstance(arr2[0], list): # uses inside length of a row
                    arr1 = [float(arr1) for _ in range(len(arr2[0]))]
                else:
                    arr1 = [float(arr1) for _ in range(len(arr2))]

            if isinstance(arr2[0], list):  # Check if arr2 is a 2D array
                if isinstance(arr1[0], list):  # For 2d x 2d
                    return [DataMethod.Multiply(row1, row2) for row1, row2 in zip(arr1, arr2)]
                else: # For 1d x 2d
                    return [DataMethod.Multiply(arr1, row) for row in arr2]
            else: # For 1d x 1d
                return [a * b for a, b in zip(arr1, arr2)]
        except Exception as ex:
            print(ex)

    # shuffles two lists while maintaining their corresponding element
    @staticmethod
    def ShuffleData(X, Y): 
        a = list(zip(X, Y))
        random.shuffle(a)
        X, Y = zip(*a)
        return list(X), list(Y)