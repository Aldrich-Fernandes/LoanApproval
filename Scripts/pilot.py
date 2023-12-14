from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from DataHandle import PreProcess

acc = []
for x in range(8):
    # Generate random data for demonstration purposes
    PreProcessor = PreProcess(New=True)
    TrainX, TrainY, TestX, TestY = PreProcessor.getData(split=0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(TrainX, TrainY, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    acc.append(accuracy)
    print(f"Accuracy: {accuracy}")

print(f"Average: {sum(acc)/len(acc)}")