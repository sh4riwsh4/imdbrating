import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_json('data.json')

# Veri setini hazırlama
df = shuffle(df, random_state=42)
X = df[['Movie Year', 'Budget', 'Gross', 'Director', 'Genres', 'Raters', 'TopActor1', 'TopActor2', 'TopActor3', 'Company1', 'Company2', 'Language']]
yRating = df['Rating']

xTrain, xTest, yTrainRating, yTestRating = train_test_split(X, yRating, test_size=0.2)

def main():
    st.title("Random Forest Regressor Modeli İle IMDB Film Rating Tahmini")

    modelRating = RandomForestRegressor(n_estimators=100, random_state=42)
    modelRating.fit(xTrain, yTrainRating)

    y_pred_rating = modelRating.predict(xTest)
    r2_rating = r2_score(yTestRating, y_pred_rating)
    mse_rating = mean_squared_error(yTestRating, y_pred_rating)
    st.subheader("Performans Metrikleri:")
    st.write("Rating - Mean Squared Error:", mse_rating)
    st.write("Rating - R2 Score:", r2_rating)

    st.subheader("Kullanıcı Verileri İle Tahminler:")
    movieYear = st.number_input("Movie Year", min_value=1900, max_value=2023)
    movieBudget = st.number_input("Budget", min_value=0, max_value=1000000000)
    movieDirector = st.text_input("Director", placeholder="Martin Scorsese")
    movieGenre1 = st.text_input("Genre 1", placeholder="Action")
    movieGenre2 = st.text_input("Genre 2", placeholder="Drama")
    movieRaters = st.number_input("Raters", min_value=0)
    movieGross = st.number_input("Gross", min_value=0)
    movieTopActor1 = st.text_input("Top Actor 1", placeholder="Johnny Depp")
    movieTopActor2 = st.text_input("Top Actor 2", placeholder= "Natalie Portman")
    movieTopActor3 = st.text_input("Top Actor 3", placeholder= "Adam Driver")
    movieCompany1 = st.text_input("Company 1", placeholder= "Lucas Films")
    movieCompany2 = st.text_input("Company 2", placeholder= "Dreamworks")
    movieLanguage = st.text_input("Language", placeholder= "English")

    dataframe = pd.DataFrame(columns=["Director", "Genres", "TopActor1", "TopActor2", "TopActor3", "Company1", "Company2",
    "Language", "Budget" , "Gross", "Raters"])

    columnsToNormalize = ['Budget', 'Gross', 'Raters']
    categoricalCols = ["Director", "Genres", "TopActor1", "TopActor2", "TopActor3", "Company1", "Company2", "Language"]

    movieYear = 2023 - movieYear
    movieGenres = ''.join(sorted([movieGenre1, movieGenre2]))

    labelEncoder = LabelEncoder()

    data = {'Director': [movieDirector],
            'Genres': [movieGenres],
            'TopActor1': [movieTopActor1],
            'TopActor2': [movieTopActor2],
            'TopActor3': [movieTopActor3],
            'Company1': [movieCompany1],
            'Company2': [movieCompany2],
            'Language': [movieLanguage],
            'Budget': [movieBudget],
            'Gross': [movieGross],
            'Raters': [movieRaters],
    }
    dataframe = pd.concat([dataframe, pd.DataFrame(data, index=[0])])

    for col in categoricalCols:
        dataframe[col] = labelEncoder.fit_transform(dataframe[col])
        dataframe[col] = dataframe[col].values

    scaler = MinMaxScaler(feature_range=(0, 100))
    df[columnsToNormalize] = scaler.fit_transform(df[columnsToNormalize])

    movieDirector = dataframe['Director'][0]
    movieGenres = dataframe['Genres'][0]
    movieTopActor1 = dataframe['TopActor1'][0]
    movieTopActor2 = dataframe['TopActor2'][0]
    movieTopActor3 = dataframe['TopActor3'][0]
    movieCompany1 = dataframe['Company1'][0]
    movieCompany2 = dataframe['Company2'][0]
    movieLanguage = dataframe['Language'][0]

    prediction = modelRating.predict([[movieYear, movieBudget, movieDirector, movieGenres, movieRaters, movieGross, movieTopActor1,
    movieTopActor2, movieTopActor3, movieCompany1, movieCompany2, movieLanguage]])[0]

    # Sonucu gösterme
    st.subheader("IMDB Rating Prediction")
    st.write(f"Prediction: {int(prediction)}/100")

if __name__ == '__main__':
    main()