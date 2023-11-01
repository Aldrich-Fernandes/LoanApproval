initialLearningRate = 1.0
decay = 0.075

for step in range(80):
    learningRate =  initialLearningRate / (1 + decay * step)
    print(learningRate)