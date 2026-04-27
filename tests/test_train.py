from src.train import train_model
import numpy as np

def test_training():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)

    model = train_model(X, X, y, y)
    assert model is not None