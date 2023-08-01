from DataHandle import DataMethod as DP

a = [["A1", "A2", "A3"],
     ["B1", "B2", "B3"]]

for row in a:
    row[0] = "XX"

print(a)