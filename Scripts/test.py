file = open(r"DataSet\Models\highAcc.txt", "r")

HiddenLayerWeights = eval(file.readline().rstrip())
HiddenLayerBiases = eval(file.readline().rstrip())
OutputLayerWeights = eval(file.readline().rstrip())
OutputLayerBiases = eval(file.readline().rstrip())
ScalingData = eval(file.readline().rstrip())

print(f"{HiddenLayerWeights}\n{type(HiddenLayerWeights)}",
      f"\n\n\n{HiddenLayerBiases}\n{type(HiddenLayerBiases)}",
      f"\n\n\n{ScalingData}\n{type(ScalingData)}",
      )
