from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression(X_train, X_test, y_train, y_test, C_val, solver):
    model = LogisticRegression(C=C_val, solver=solver, max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return acc, pred
