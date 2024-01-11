from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DataHandle import PreProcess

PreProcessor = PreProcess()
LogRegAcc = []
RandForestAcc = []

def LogReg(X_train, X_test, y_train, y_test):
    acc = []
    for _ in range(10):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc.append(accuracy)

    return round(sum(acc)/len(acc), 4)

def RandForest(X_train, X_test, y_train, y_test):
    acc = []
    for _ in range(10):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        acc.append(accuracy)

    return round(sum(acc)/len(acc), 4)

def testModels():
    for x in range(10):
        print(f"Training dataset {x+1}")
        PreProcessor.newDataset()
        TrainX, TrainY, _, _ = PreProcessor.getData(split=0)
        X_train, X_test, y_train, y_test = train_test_split(TrainX, TrainY, test_size=0.2, random_state=42)

        LogRegAcc.append(LogReg(X_train, X_test, y_train, y_test))
        RandForestAcc.append(RandForest(X_train, X_test, y_train, y_test))

    print(f"Logistic Regression     | Accuracy: {round(sum(LogRegAcc)/len(LogRegAcc), 4)}")
    print(f"Random Forest           | Accuracy: {round(sum(RandForestAcc)/len(RandForestAcc), 4)}")

testModels()