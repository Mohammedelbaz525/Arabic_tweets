from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import csv
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score , StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics , svm
from sklearn import linear_model
import seaborn as sns
from sklearn import  datasets
from nltk.corpus import stopwords
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix, classification_report


def import_data():
    # import total dataset
    df = pd.read_csv('tweets.csv')

    df.columns = ['label', 'tweets']
    df = df.sample(frac=1)

    # set stopwords
    stop_words = set(stopwords.words('arabic'))

    # Remove stop words
    df['tweets'] = df['tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # clean data
    df['tweets'] = df['tweets'].str.replace(r'[^ \t ا-ي ]', '')

    # get a list of column names
    headers = list(df.columns.values)

    # separate into independent and dependent variables
    x = df[headers[0,:]]
    y = df[headers[:,0]].values.ravel()

    return x, y


if __name__ == '_main_':    # get training and testing sets
    x, y = import_data()

    # set to 20 folds
    kf = KFold(K=20)

    # blank lists to store predicted values and actual values
    predicted_y = []
    expected_y = []

    # partition data
    for train_index, test_index in kf.split(x, y):
        # specific ".loc" syntax for working with dataframes
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # create and fit classifier
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)

        # store result from classification
        predicted_y.extend(classifier.predict(x_test))

        # store expected result for this specific fold
        expected_y.extend(y_test)

    # save and print accuracy
    accuracy = metrics.accuracy_score(expected_y, predicted_y)
    print("Accuracy: " + accuracy._str_())