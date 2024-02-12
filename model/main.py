import numpy as np 
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def create_model():
    # Load the data
    data = get_clean_data()

    # Split the data into features and target
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

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
    
    model = create_model()


if __name__ == "__main__":
    main()