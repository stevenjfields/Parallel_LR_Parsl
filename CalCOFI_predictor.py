# IMPORTANT! I used the initial data prep from here: https://www.kaggle.com/code/jonasnconde/data-cleaning-challenge-scale-and-normalize-data
# The goal of this project is to test parallelizing hyper-parameter tuning, not creating new models or EDA.

from ensurepip import bootstrap
import numpy as np 
import pandas as pd 
import itertools
import parsl
from scipy import stats
from mlxtend.preprocessing import minmax_scaling
from parsl import python_app
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import timeit
from configs.local import local_threading

parsl.clear()
parsl.load(local_threading(6))

def data_prep():
    calcofi = pd.read_csv("./datasets/bottle.csv")
    calcofi_subset = calcofi.loc[:,'Depthm':'O2Satq']
    calcofi_subset = calcofi_subset.fillna(calcofi_subset.mean())
    print(calcofi_subset.describe())
    print(calcofi_subset.info())

    x = calcofi_subset.drop("Salnty", axis=1)
    y = calcofi_subset.Salnty

    return train_test_split(x, y, random_state=0)


def build_parameter_lists():
    # Shared hyper-parameters
    alpha = [0.1, 0.5, 1., 2., 5.]
    max_iter = [100, 1000, 10000]
    tol = [0.0001, 0.001, 0.01]
    num_jobs = [-1]

    # Lasso hyper-parameters
    lasso_selection = ["cyclic", "random"]

    # Ridge hyper-parameters
    ridge_solver = ['auto']

    # Elasticnet
    l1_ratio = [0.1, 0.25, 0.5, 1.]

    # Polynomial Regression
    degree = [i for i in range(5)]

    # Decision Tree Regression
    # Took out MAE error because it took forever
    criterion = ["squared_error", "friedman_mse", "poisson"]
    splitter = ["best"]

    # Ramdom Forest Regression
    n_estimators = [2, 5, 10, 25, 50]
    bootstrap = [True]

    baseline_lr = itertools.product(num_jobs)
    lasso_list = itertools.product(num_jobs, alpha, max_iter, tol, lasso_selection)
    ridge_list = itertools.product(num_jobs, alpha, max_iter, tol, ridge_solver)
    elastic_list = itertools.product(num_jobs, alpha, max_iter, tol, l1_ratio)
    polynomal_regression = itertools.product(num_jobs, degree)
    # num_jobs will get ignored but it's easier to include than rewrite train_lr_model again
    decision_tree_regression = itertools.product(num_jobs, criterion, splitter)
    random_forest_regression = itertools.product(num_jobs, n_estimators, criterion, bootstrap)

    param_set = list()

    for i in baseline_lr:
        param_set.append(("baseline_lr", i))
        pass
    for i in lasso_list:
        param_set.append(("lasso", i))
        pass
    for i in ridge_list:
        param_set.append(("ridge", i))
        pass
    for i in elastic_list:
        param_set.append(("elastic", i))
        pass
    for i in polynomal_regression:
        #param_set.append(("polynomial_lr", i))
        pass
    for i in decision_tree_regression:
        param_set.append(("dt_regression", i))
    for i in random_forest_regression:
        param_set.append(("rf_regression", i))
    
    return param_set

@python_app
def train_lr_model(model, parameters, x_train, y_train):
    num_jobs = parameters[0]
    if model in ["lasso", "ridge", "elastic"]:
        alpha = parameters[1]
        max_iter = parameters[2]
        tol = parameters[3]

    lr_model = None
    if model == "baseline_lr":
        lr_model = LinearRegression(n_jobs=num_jobs)
    elif model == "lasso":
        selection = parameters[4]
        lr_model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, selection=selection)
    elif model == "ridge":
        solver = parameters[4]
        lr_model = Ridge(alpha=alpha, max_iter=max_iter, tol=tol, solver=solver)
    elif model == "elastic":
        l1_ratio = parameters[4]
        lr_model = ElasticNet(alpha=alpha, max_iter=max_iter, tol=tol, l1_ratio=l1_ratio)
    elif model == "polynomial_lr":
        degree = parameters[1]
        poly_df = PolynomialFeatures(degree)
        x_train = poly_df.fit_transform(x_train)
        lr_model = LinearRegression(n_jobs=num_jobs)
    elif model == "dt_regression":
        criterion = parameters[1]
        splitter = parameters[2]
        lr_model = DecisionTreeRegressor(criterion=criterion, splitter=splitter)
    elif model == "rf_regression":
        n_estimators = parameters[1]
        criterion = parameters[2]
        bootstrap = parameters[3]
        lr_model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, bootstrap=bootstrap, n_jobs=num_jobs)
    else:
        print("Model not found")
        return

    lr_model.fit(x_train, y_train)
    return lr_model

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = data_prep()

    model_param_list = build_parameter_lists()

    model_futures = [train_lr_model(i[0], i[1], x_train, y_train) for i in model_param_list]

    columns = ["train_score", "test_score", "model", "arg_list"]
    results = []

    start = timeit.default_timer()
    i = 0
    for model_future in model_futures:
        if i % 5 == 0:
            print(f"Trained {i} linear regression models.")
            print("Savings results ...")
            pd.DataFrame(results).to_csv("CalCOFI_results.csv")
        #print(model_future.task_def["args"][1])
        model = model_future.result()

        if model_param_list[i][0] == "polynomial_lr":
            poly = PolynomialFeatures(degree=model_param_list[i][1][1])
            poly_train = poly.fit_transform(x_train)
            poly_test = poly.transform(x_test)

            train_score = model.score(poly_train, y_train)
            test_score = model.score(poly_test, y_test)
        else:
            train_score = model.score(x_train, y_train)
            test_score = model.score(x_test, y_test)
        
        score_values = {
            columns[0]: train_score,
            columns[1]: test_score,
            columns[2]: model_future.task_def["args"][0],
            columns[3]: model_future.task_def["args"][1]
        }

        i += 1
        results.append(score_values)
        print(score_values)

    stop = timeit.default_timer()

    results = pd.DataFrame(results)
    best_train = results["train_score"].idxmax()
    best_test = results["test_score"].idxmax()

    print(f"Best model; train: {results.iloc[best_test, 0]}, test: {results.iloc[best_test, 1]}")
    print(f"Model parameters: {model_param_list[best_test]}")
    print(f"Execution time: {stop-start} seconds.")