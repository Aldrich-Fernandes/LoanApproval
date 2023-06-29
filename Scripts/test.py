from numpy import array
data = [   
['5008806', 'M', '1', '1', '0', '112500', 'Working ', 'Secondary / secondary special   ', 'Married ', 'House / apartment   ', '1', '0', '0', '0', 'Security staff', '2', '59', '4', '0', '30', '1'],
['5008808', 'F', '0', '1', '0', '270000', 'Commercial associate', 'Secondary / secondary special   ', 'Single / not married', 'House / apartment   ', '1', '0', '1', '1', 'Sales staff   ', '1', '53', '9', '0', '5', '1'],
['5008809', 'F', '0', '1', '0', '270000', 'Commercial associate', 'Secondary / secondary special   ', 'Single / not married', 'House / apartment   ', '1', '0', '1', '1', 'Sales staff   ', '1', '53', '9', '0', '5', '1'],
['5008810', 'F', '0', '1', '0', '270000', 'Commercial associate', 'Secondary / secondary special   ', 'Single / not married', 'House / apartment   ', '1', '0', '1', '1', 'Sales staff   ', '1', '53', '9', '0', '27', '1'],
['5008811', 'F', '0', '1', '0', '270000', 'Commercial associate', 'Secondary / secondary special   ', 'Single / not married', 'House / apartment   ', '1', '0', '1', '1', 'Sales staff   ', '1', '53', '9', '0', '39', '1']]

FeatureColumns = []
for y in range(len(data[0])):
    FeatureColumns.append([data[x][y].strip() for x in range(len(data))])

print("\n\n",FeatureColumns)
for index, features in enumerate(FeatureColumns):
    try: # For intergers
        FeatureColumns[index] = list(map(int, features))
    except:
        uniqueStrings = {}
        featureIndex = 0
        val = 0
        for uniqueString in features:
            if uniqueString not in uniqueStrings.keys():
                uniqueStrings[uniqueString] = val
                val += 1
            
            features[featureIndex] = uniqueStrings[uniqueString]
            featureIndex += 1

print("\n\n",FeatureColumns)
print("\n\n", FeatureColumns)
