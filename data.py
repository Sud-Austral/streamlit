import pandas as pd
def get_data():
    df = pd.read_excel("prueba_minimax.xlsx")
    df["region"] = df["Región"]
    return df

