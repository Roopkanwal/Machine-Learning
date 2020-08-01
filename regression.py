# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% SalePrice distribution
import seaborn as sns

sns.distplot(train["SalePrice"])


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc

sns.boxplot(x="CentralAir", y="SalePrice", data=train)

# %%
sns.boxplot(x="BldgType", y="SalePrice", data=train)

# %%

sns.boxplot(x="OverallQual", y="SalePrice", data=train)

# %% SalePrice distribution w.r.t Neighborhood

import matplotlib.pyplot as plt 

plt.figure(figsize=(16,8))
ax = sns.boxplot(x="Neighborhood", y="SalePrice", data=train)
ax.set_xticklabels(
    ax.get_xticklabels(), rotation= 45
)
# %% SalePrice distribution w.r.t YearBuilt

plt.figure(figsize=(35,15))
built_after_2000=train["YearBuilt"]>2000
ax = sns.boxplot(x="YearBuilt", y="SalePrice", data=train[built_after_2000])
ax.set_xticklabels(
    ax.get_xticklabels(), rotation= 45, fontsize=15
)

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.2f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")


# %% your solution to the regression problem
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

numeric_columns=[
    "YrSold",
    "GarageYrBlt",
    "1stFlrSF",
    "2ndFlrSF",
    "BsmtFullBath",
    "FullBath"]

cat_col= ["Neighborhood","SaleCondition","Heating","BldgType","Street"]               
selected_columns =numeric_columns+cat_col

train_x = train[selected_columns]
train_y = train["SalePrice"]

#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

enc= OneHotEncoder(handle_unknown='ignore')
imp=SimpleImputer()

ct = ColumnTransformer(
[
("ohe", enc, cat_col),
("fillna",imp,numeric_columns),
],
remainder="passthrough"
)

train_x = ct.fit_transform(train_x)

reg.fit(train_x, train_y)


# %%

print("Training Set Performance")
evaluate(reg, train_x, train_y)

#%%


truth = pd.read_csv("truth_house_prices.csv")
test_x = test[selected_columns]
test_y = truth["SalePrice"]

test_x=ct.transform(test_x)

print("Test Set Performance")
print(evaluate(reg, test_x, test_y))

# %%
