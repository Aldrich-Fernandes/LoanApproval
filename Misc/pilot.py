# Models used
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Unitilites
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# From my programs
from DataHandle.Preprocess import Preprocess

# random forest classification - utilises tree traversal
def RandForest(X_train, X_test, y_train, y_test):
    acc = []
    for _ in range(10):     # Generates 10 different models to get average acurracy.
        # Initialising a model
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Training the model
        model.fit(X_train, y_train)

        # Testing performace
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        acc.append(accuracy)

    return round(sum(acc)/len(acc), 4)

# Logistic regression model - ultilies neural networks
def LogReg(X_train, X_test, y_train, y_test):   # Same as RanForest()
    acc = []
    for _ in range(10):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        acc.append(accuracy)

    return round(sum(acc)/len(acc), 4)

def testModels():        
    Preprocessor = Preprocess()
    LogRegAcc = []
    RandForestAcc = []

    for x in range(10): # Trains models with 10 different datasets.
        print(f"Training dataset {x+1}")
        Preprocessor.newDataset()
        TrainX, TrainY, _, _ = Preprocessor.getData(split=0)
        X_train, X_test, y_train, y_test = train_test_split(TrainX, TrainY, test_size=0.2, random_state=42)

        LogRegAcc.append(LogReg(X_train, X_test, y_train, y_test))
        RandForestAcc.append(RandForest(X_train, X_test, y_train, y_test))

    # Average accuracy after 100 tests each. 
    print(f"Logistic Regression     | Accuracy: {round(sum(LogRegAcc)/len(LogRegAcc), 4)}")
    print(f"Random Forest           | Accuracy: {round(sum(RandForestAcc)/len(RandForestAcc), 4)}")

testModels()