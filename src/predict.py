import joblib

model = joblib.load("models/model.pkl")

def predict(data):
    return model.predict([data])