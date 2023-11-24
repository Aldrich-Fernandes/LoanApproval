from DataHandle import DataMethod as DM

a = -2

b = [1, 2, 4, 2, 1]

c = [[1, 2, 3, 1, 4],
     [4, 9, 4, 3, 2],
     [7, 4, 3, 7, 2]]

d = [[3],
     [1],
     [4],
     [6],
     [1]]

print("0x1")
print(DM.Multiply(a, b))
print("0x2")
print(DM.Multiply(a, c))
print("1x1")
print(DM.Multiply(b, b))
print("1x2")
print(DM.Multiply(b, c))
print("2x2")
[print(x) for x in DM.Multiply(c, c)]