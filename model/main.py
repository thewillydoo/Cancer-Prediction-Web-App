import numpy as np 
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def get_clean_data():
    # Load the data
    data = pd.read_csv('data/data.csv')

    # Drop the columns that are not required
    data = data.drop(['id', 'Unnamed: 32'], axis=1)

    # Convert the diagnosis column to binary
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


    return data


def main():

    data = get_clean_data()
    print(data.head())


if __name__ == "__main__":
    main()