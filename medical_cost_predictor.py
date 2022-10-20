import numpy as np 
import pandas as pd 
import itertools
import parsl
from parsl import python_app
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split

from configs.local import local_threading
parsl.clear()
parsl.load(local_threading(1))


def data_prep():
    data = pd.read_csv('./datasets/insurance.csv')

    #sex
    le = LabelEncoder()
    le.fit(data.sex.drop_duplicates()) 
    data.sex = le.transform(data.sex)
    # smoker or not
    le.fit(data.smoker.drop_duplicates()) 
    data.smoker = le.transform(data.smoker)
    #region
    le.fit(data.region.drop_duplicates()) 
    data.region = le.transform(data.region)
    
    return data

def build_parameter_lists():
    # Shared hyper-parameters
    alpha = [0.1, 0.5, 1., 2., 5.]
    #fit_intercept = [True, False]
    max_iter = [100, 1000, 10000]
    tol = [0.0001, 0.001, 0.01]

    # Lasso hyper-parameters
    lasso_selection = ["cyclic", "random"]

    # Ridge hyper-parameters
    ridge_solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

    # Elasticnet
    l1_ratio = [0.1, 0.25, 0.5, 1.]

    lasso_list = itertools.product(alpha, max_iter, tol, lasso_selection)
    ridge_list = itertools.product(alpha, max_iter, tol, ridge_solver)
    elastic_list = itertools.product(alpha, max_iter, tol, l1_ratio)

    param_set = list()

    for i in lasso_list:
        param_set.append(("lasso", i))
    for i in ridge_list:
        param_set.append(("ridge", i))
    for i in elastic_list:
        param_set.append(("elastic", i))
    
    return param_set

@python_app
def train_lr_model(model, parameters, x_train, y_train):
    alpha = parameters[0]
    max_iter = parameters[1]
    tol = parameters[2]

    lr_model = None
    if model == "lasso":
        selection = parameters[3]
        lr_model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, selection=selection)
    elif model == "ridge":
        solver = parameters[3]
        lr_model = Ridge(alpha=alpha, max_iter=max_iter, tol=tol, solver=solver)
    elif model == "elastic":
        l1_ratio = parameters[3]
        lr_model = ElasticNet(alpha=alpha, max_iter=max_iter, tol=tol, l1_ratio=l1_ratio)
    else:
        print("Model not found")
        return

    lr_model.fit(x_train, y_train)
    return lr_model

if __name__ == '__main__':
    data = data_prep()

    x = data.drop(['charges'], axis = 1)
    y = data.charges

    x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)

    model_param_list = build_parameter_lists()
    
    model_futures = [train_lr_model(i[0], i[1], x_train, y_train) for i in model_param_list]

    columns = ["train_score", "test_score"]
    results = pd.DataFrame(columns=columns)

    i = 0
    for model_future in model_futures:
        model = model_future.result()
        score_values = { 
            columns[0]: model.score(x_train, y_train), 
            columns[1]: model.score(x_test, y_test)    
        }
        scores = pd.DataFrame(score_values, index=[i])

        i+=1
        results = pd.concat([results, scores])

    best_train = results["train_score"].idxmax()
    best_test = results["test_score"].idxmax()

    print(f"Best model; train: {results.iloc[best_test, 0]}, test: {results.iloc[best_test, 1]}")
    print(f"Model parameters: {model_param_list[best_test]}")