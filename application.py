import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib 

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

    # cls = LogisticRegression(solver='sag', n_jobs=4) # slow
    cls = LinearRegression()
    # cls = RandomForestRegressor() # slow
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("poly", PolynomialFeatures(degree=2)), # this increases score from ~.2 to ~.42; degree=3 fails due to memory requirement. takes a few mins
            ("classifier", cls),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 15)
    clf.fit(x_train, y_train)
    
    print("model score: %.3f" % clf.score(x_test, y_test))

    joblib.dump(clf, f"model.joblib")    

def test():
    clf = joblib.load(f'model.joblib')
    cities = ['Kachidoki', 'Shibuya', 'Seijogakuenmae', 'Monzennakacho', 'Iwamotocho', 'Kamiitabashi', 'Roppongi']

    for city in cities:
        d = {
            'NearestStation': [city],
            'TimeToNearestStation': [10],
            'Area': [80],
            'BuildingYear': [2010],
            'Year': [2020],
            'Type': ['"Pre-owned Condominiums, etc."'],
            'FloorPlan': ['3LDK'], 
            'Renovation': ['Not yet'],
        }
        df = pd.DataFrame(data=d)
        df
        p = clf.predict(df)
        print(f'{city}: {p/100000}')

    # The above gives the following output, which should represent the cost of housing
    # in each of the cities. But...it's clearly not correct, nor accurate even
    # relative to each other (e.g., Shibuya should not be cheaper than Seijogakuenmae)        
    #   Kachidoki: [-1673.41200719]
    #   Shibuya: [454.42983849]
    #   Seijogakuenmae: [793.22483103]
    #   Monzennakacho: [999.29303348]
    #   Iwamotocho: [1250.53065929]
    #   Kamiitabashi: [806.85889658]
    #   Roppongi: [751.41300991]
    
#run()
test()



