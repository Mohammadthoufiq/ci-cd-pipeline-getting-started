from src.data import load_data

def test_load_data():
    df = load_data("data/creditcard.csv")

    # Check if data loaded
    assert df is not None

    # Check if not empty
    assert len(df) > 0

    # Check fraud column exists
    assert "Class" in df.columns