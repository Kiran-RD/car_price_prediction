# To add a new cell, type '### %%'
# To add a new markdown cell, type '### %% [markdown]'
# %%
import pickle
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
data = pd.read_csv("car data.csv")
data.head()


# %%
data.describe()


# %%
d = {}
for col in data.columns:
    d[col] = len(data[col].unique())
d


# %%
data.isnull().sum()


# %%
sns.pairplot(data.iloc[:, [1, 2, 3, 4]])


# %%
data_processed = data.drop(['Car_Name'], axis=1)
data_processed['age'] = 2021-data_processed.Year
data_processed = data_processed.drop(['Year'], axis=1)


# %%
data_processed.head()


# %%
data_processed = pd.get_dummies(data_processed, drop_first=True)


# %%
data_processed


# %%
data_processed.corr()


# %%
sns.pairplot(data_processed)


# %%
corrmat = data_processed.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(data_processed.corr(), annot=True, cmap='RdYlGn')


# %%
data_processed.head()


# %%
y = data_processed.Selling_Price
x = data_processed.iloc[:, 1:]


# %%
model = ExtraTreesRegressor()
model.fit(x, y)


# %%
model.feature_importances_


# %%
imp_features = pd.Series(model.feature_importances_,
                         index=x.columns).sort_values()
imp_features.plot(kind='barh')


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# %%
x_train.shape


# %%
rf_random = RandomForestRegressor()


# %%
# HyperParameters
n_estimators = [int(m) for m in np.linspace(start=100, stop=1200, num=12)]
n_estimators


# %%

# Randomized Search CV
# Helps in finding right Hyperparameter values

# Number of trees in random forest
n_estimators = [int(m) for m in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(m) for m in np.linspace(5, 30, num=6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# %%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()


# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                               scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)


# %%
rf_random.fit(x_train, y_train)


# %%
predictions = rf_random.predict(x_test)


# %%
sns.distplot(y_test-predictions)


# %%
plt.scatter(y_test, predictions)


# %%
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# %%

# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)  # used for deployment
