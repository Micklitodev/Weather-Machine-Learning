import pandas as pd
import numpy as np
import time
from scipy import stats
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def loadCSV(file_path):
    df = pd.read_csv(file_path)
    print(f"Weather data frame is: {df.shape}")

    df.drop(
        columns=["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am", "Location", "Date"],
        inplace=True,
    )
    df = df.dropna(how="any")

    z = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z < 3).all(axis=1)]

    df["RainToday"].replace({"No": 0, "Yes": 1}, inplace=True)
    df["RainTomorrow"].replace({"No": 0, "Yes": 1}, inplace=True)

    df = pd.get_dummies(df, columns=["WindGustDir", "WindDir3pm", "WindDir9am"])

    return df


def formatData(df):
    scaler = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df), index=df.index, columns=df.columns
    )
    return df_scaled


def feature_selection(X, y, k):
    selector = SelectKBest(chi2, k=k)
    selector.fit(X, y)
    X_new = selector.transform(X)
    print("Selected Features:", X.columns[selector.get_support(indices=True)])
    return X_new


def logReg (X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    t0 = time.time()
    clf_logreg = LogisticRegression(random_state=0)
    clf_logreg.fit(X_train, y_train.values.ravel())

    y_pred = clf_logreg.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print("LR Accuracy:", score)
    print("TimeComplexity:", time.time() - t0)


## Main Exe
if __name__ == "__main__":
    df = loadCSV("./weatherAUS.csv")
    df = formatData(df)

    X = df.loc[:, df.columns != "RainTomorrow"]
    y = df[["RainTomorrow"]]
    X_new = feature_selection(X, y, k=3)
    logReg(X_new, y)
