import pandas as pd

def load_and_merge_data():
    business = pd.read_csv("../data/business.csv")
    economy = pd.read_csv("../data/economy.csv")

    business['Class'] = 'Business'
    economy['Class'] = 'Economy'

    df = pd.concat([business, economy], ignore_index=True)
    return df

if __name__ == "__main__":
    df = load_and_merge_data()
    print("Loaded dataset shape: ", df.shape)
    print("Columns: ", df.columns)
    print(df.head())
