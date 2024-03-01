from Scripts.DataHandle import DataMethod

import unittest

class TestDataMethods(unittest.TestCase):
    def test_CsvToArray(self):
        path = "DataSet\HomeLoanTrain.csv"
        
        # Call the method under test
        result = DataMethod.CsvToArray(path, maxEntries=10)
        
        # Assert the expected result
        self.assertIsInstance(result, list)
        self.assertNotIn('', result)

    def test_Transpose(self):
        array = [[2, 4, 3], [5, 6, 7]]
        
        expected = [[2, 5], [4, 6], [3, 7]]
        
        result = DataMethod.Transpose(array)

        self.assertListEqual(expected, result)

    def test_DotProduct(self):
        arr1 = [[1, 2], [3, 4], [5, 6]]
        
        arr2 = [[7, 8, 9], [10, 11, 12]]
        
        expected_result = [[27, 30, 33], 
                           [61, 68, 75], 
                           [95, 106, 117]]
        
        result = DataMethod.DotProduct(arr1, arr2)
        self.assertListEqual(result, expected_result)

    def test_Multiply(self):
        tests = {
            "Test1": {"scalar": 2,
                      "array": [1, 2, 3, 4, 5],
                      "expected": [2, 4, 6, 8, 10]},
            "Test2": {"scalar": 2,
                      "array": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      "expected": [[2, 4, 6], [8, 10, 12], [14, 16, 18]]},
            "Test3": {"scalar": [1, 2, 3],
                      "array": [4, 5, 6],
                      "expected": [4, 10, 18]}
                           }
        
        for test in tests.values():
            result = DataMethod.Multiply(test["scalar"], test["array"])
            self.assertListEqual(result, test["expected"])

