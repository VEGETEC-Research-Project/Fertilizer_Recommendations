import pandas as pd

def preprocess_input(data, expected_columns):
    # Preprocess input data (convert crop type and align features) 
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['Crop_Type'], drop_first=True)
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df
