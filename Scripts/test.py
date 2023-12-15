class DataMethod:
    @staticmethod
    def Transpose(array):
        return [[array[x][y] for x in range(len(array))] for y in range(len(array[0]))]
    
    @staticmethod
    def DotProduct(arr1, arr2):
        if not isinstance(arr1, list):
            arr1 = [float(arr1) for _ in range(len(arr2[0]))]
        elif not isinstance(arr2, list):
            arr2 = [[float(arr2)] for _ in range(len(arr1))]

        if isinstance(arr2[0], list):
            if isinstance(arr1[0], list):
                # For 2D x 2D
                return [DataMethod.DotProduct(row1, row2) for row1, row2 in zip(arr1, arr2)]
            else:
                # For 1D x 2D
                return [DataMethod.DotProduct(arr1, row) for row in zip(*arr2)]
        else:
            # For 1D x 1D
            return sum(a * b for a, b in zip(arr1, arr2))


# Example usage:
arr1 = [[1, 2], [3, 4]]
arr2 = [[5, 6], [7, 8]]

result = DataMethod.DotProduct(arr1, arr2)
print(result)