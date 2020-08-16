from AER import AER
from os import listdir
from os.path import join
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import hyper_param_eval as hpe

GLOBAL_RANDOM_STATE = 2020
N_FOLDS = 10
xgb_params = {
    'eval_metric': 'error',
    'max_depth': 50,
    'objective': 'binary:logistic',
    'random_state': GLOBAL_RANDOM_STATE
}

dt_params = {
    'max_depth': 50,
    'min_samples_split': 10,
    'random_state': GLOBAL_RANDOM_STATE
}

params = {
    'cov_type': 'tied',
    'lambda_A': 0.6,
    'max_steps': 5,
    'max_loss': 0.01,
    'K': 5,
    'xgb_params': xgb_params,
    'random_state': GLOBAL_RANDOM_STATE
}

Results = []


def perf_measure(test_y, preds):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    test_y = test_y.to_list()

    for i in range(len(preds)):
        if test_y[i] == preds[i] == 1:
            TP += 1
        if preds[i] == 1 and test_y[i] != preds[i]:
            FP += 1
        if test_y[i] == preds[i] == 0:
            TN += 1
        if preds[i] == 0 and test_y[i] != preds[i]:
            FN += 1

    return TP, FP, TN, FN


def eval_model(dev_X, dev_y, test_X, test_y, model, data_set_name, fold_n, alg_name, params):
    start_fit = datetime.now()
    model.fit(dev_X, dev_y)
    end_fit = datetime.now()
    preds = model.predict(test_X)
    end_pred = datetime.now()
    accuracy = accuracy_score(test_y, preds)
    if len(np.unique(test_y)) == 1 or len(np.unique(preds)) == 1:
        auc = None
        prs = None
    else:
        prs = average_precision_score(test_y, preds)
        auc = roc_auc_score(test_y, preds)
    TP, FP, TN, FN = perf_measure(test_y, preds)
    if TP + FN == 0:
        TPR = None
    else:
        TPR = TP / (TP + FN)
    if FP + TN == 0:
        FPR = None
    else:
        FPR = FP / (FP + TN)
    precision = precision_score(test_y, preds)
    fit_time = (end_fit - start_fit).total_seconds()
    inference = (end_pred - end_fit).total_seconds() / len(test_y) * 1000
    Results.append([data_set_name, alg_name, fold_n, params, accuracy,
                    TPR, FPR, precision, auc, prs, fit_time, inference])


def eval_other_models(dev_X, dev_y, test_X, test_y, data_set_name, fold_n):
    xgb_model = xgb.XGBClassifier(**xgb_params)
    dtc_model = DecisionTreeClassifier(**dt_params)

    eval_model(dev_X, dev_y, test_X, test_y, xgb_model, data_set_name, fold_n, 'XGBoost', xgb_params)
    eval_model(dev_X, dev_y, test_X, test_y, dtc_model, data_set_name, fold_n, 'Decision Tree', dt_params)


def prepare_data(path, feature_type='min'):
    train = pd.read_csv(path)
    train.head()

    target_name = train.columns[-1]
    minority_class = choose_feature(train, target_name, feature_type)
    y = train[target_name]
    y = (y != minority_class).replace({True: 1, False: 0})
    X = train.drop(columns=[target_name])
    df_num = X.select_dtypes(exclude=[np.number])

    if len(df_num) > 0:
        df_num = df_num.fillna("None")
        enc = OneHotEncoder().fit(df_num)
        X = pd.concat([X, pd.DataFrame(enc.transform(df_num).toarray())], axis=1)
        X.drop(columns=df_num.columns, inplace=True)

    X = X.fillna(-1)
    return X, y


def choose_feature(train, target_name, feature_type):
    if feature_type == 'min':
        return train[target_name].value_counts().idxmin()
    else:
        return train[target_name].value_counts().idxmax()


def evaluate(optimize, feature_type, path, data_set_name):
    X, y = prepare_data(path, feature_type)
    if (y == 0).sum() < 0.05 * len(y):
        print("Too few 0 targets. Skipping this data set!")
        return
    kf = KFold(n_splits=N_FOLDS, random_state=GLOBAL_RANDOM_STATE, shuffle=True)
    fold_splits = kf.split(X, y)
    i = 0
    for dev_index, test_index in fold_splits:
        i += 1
        print("{}: start {} fold out of {}".format(datetime.now(), i, N_FOLDS))
        dev_X, test_X = X.iloc[dev_index], X.iloc[test_index]
        dev_y, test_y = y.iloc[dev_index], y.iloc[test_index]

        if optimize:
            distributions = dict(max_steps=[2, 5, 8, 10, 20], K=[3, 5, 7, 10, 20], cov_type=['full', 'tied', 'diag', 'spherical'])
            chosen_params = hpe.get_optimized_params(dev_X, dev_y, params, distributions, random_state=GLOBAL_RANDOM_STATE)
        else:
            chosen_params = params

        aer = AER(**chosen_params)
        eval_model(dev_X, dev_y, test_X, test_y, aer, data_set_name, i, 'AER', chosen_params)
        eval_other_models(dev_X, dev_y, test_X, test_y, data_set_name, i)


def getAllPaths(path):
    return [f for f in listdir(path)]


def evaluateFiles(optimize, feature_type, path, files_n=-1):
    files = getAllPaths(path)
    if files_n == -1:
        files_n = len(files)
    for i in tqdm(range(min(files_n, len(files)))):
        print('________________________________________________________')
        print('Evaluating file: {}'.format(files[i]))
        print('________________________________________________________')
        evaluate(optimize, feature_type, join(path, files[i]), files[i].split('.')[0])


if __name__ == '__main__':
    evaluateFiles(True, 'min', 'C:\\Users\\Radune\\PHD\\courses\\lemida hishuvit\\exe4\\classification_datasets')
    results_df = pd.DataFrame(Results, columns=['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]',
                                                'Hyper-Parameters Values',
                                                'Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve',
                                                'Training Time', 'Inference Time'])
    results_df.to_csv('C:\\Users\\Radune\\PHD\\courses\\lemida hishuvit\\exe4\\results.csv', index=False)
