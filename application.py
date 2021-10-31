import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def isInt(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

def run():    
    housing_data = pd.read_csv("Housing_data.csv")
    housing_data = housing_data[(housing_data['TimeToNearestStation'].astype('str').str.isnumeric())]
    housing_data['TimeToNearestStation'] = housing_data['TimeToNearestStation'].apply(int)
    housing_data.shape
    housing_data.info()
    housing_data.describe()

    x = housing_data[['Type', 'NearestStation', 'TimeToNearestStation', 'FloorPlan', 'Area', 'BuildingYear','Year', 'Renovation']]
    y = housing_data['TradePrice']

    numeric_features = ['TimeToNearestStation', 'Area', 'BuildingYear','Year']
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ['Type', 'NearestStation', 'FloorPlan', 'Renovation']
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    cls = LogisticRegression(solver='sag', n_jobs=4)
    cls = LinearRegression()
    cls = RandomForestRegressor()
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", cls)]
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    clf.fit(x_train, y_train)
    print("model score: %.3f" % clf.score(x_test, y_test))
    
run()




