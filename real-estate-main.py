#!/usr/bin/env python3

import os
import tarfile
from six.moves import urllib
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path, filter='data')
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

if __name__ == '__main__':

    fetch_housing_data()
    housing = load_housing_data()

    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_trainSet = housing.loc[train_index]
        strat_testSet = housing.loc[test_index]

    for set_ in (strat_trainSet, strat_testSet):
        set_.drop("income_cat", axis=1, inplace=True)
    
    housing = strat_trainSet.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                    s=housing["population"]/100, label="population", figsize=(10,7),
                    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                )
    plt.legend()
    # plt.show()

    housing["houshold_population"] = housing["population"] / housing["households"]
    housing["beds_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["household_rooms"] = housing["total_rooms"] / housing["households"]

    corr_matrix = housing.drop("ocean_proximity", axis=1).corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    # scatter_matrix(housing[attributes], figsize=(10, 6))
    
    # ╔══════════════════════════════════════════════════════════╗
    # ║                     PREPARING THE DATA                   ║
    # ╚══════════════════════════════════════════════════════════╝

    housing = strat_trainSet.drop("median_house_value", axis=1)
    housing_labels = strat_trainSet.copy("median_house_value")

    imputer = SimpleImputer(strategy="median")
    housing_Numeric = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_Numeric)

    print(imputer.statistics_)

    
    