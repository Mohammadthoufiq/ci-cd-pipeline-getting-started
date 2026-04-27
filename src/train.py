import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, X_test, y_train, y_test):

    with mlflow.start_run():

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        #  MLflow logging
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, "models/model.pkl")

    return model