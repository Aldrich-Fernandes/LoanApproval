from DataHandle import DataMethod

def dot_product(matrix1, matrix2):
    if isinstance(matrix1[0], list) and isinstance(matrix2[0], list):
        # Matrix dot product
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Matrix dimensions are not compatible for dot product.")

        result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]

        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]

    elif not isinstance(matrix1[0], list) and not isinstance(matrix2[0], list):
        # Vector dot product
        if len(matrix1) != len(matrix2):
            raise ValueError("Vector dimensions are not compatible for dot product.")

        result = sum(a * b for a, b in zip(matrix1, matrix2))

    else:
        raise ValueError("Input types are not compatible for dot product.")

    return result

# Example usage:
matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6, 9], [7, 8, 7]]

result_matrix = dot_product(matrix_a, matrix_b)
DMMat = DataMethod.DotProduct(matrix_a, matrix_b)

print(result_matrix)
print(DMMat)
