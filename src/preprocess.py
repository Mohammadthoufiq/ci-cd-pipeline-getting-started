from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2)