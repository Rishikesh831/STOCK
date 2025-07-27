import pandas as pd

def load_input_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    return df 