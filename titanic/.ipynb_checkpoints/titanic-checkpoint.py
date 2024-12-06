import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv("test.csv")
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    
    X = pd.get_dummies(train_data[features])
    y = train_data["Survived"]
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    
    print("Score on training data: ", model.score(X, y))
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('output.csv', index=False)