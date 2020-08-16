from AER import AER
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import KFold

N_ITER = 50
N_FOLDS = 3


def get_optimized_params(X, y, params, distributions, random_state=2020):
    kf = KFold(n_splits=N_FOLDS, random_state=random_state, shuffle=True)
    fold_splits = kf.split(X, y)

    aer = AER(**params)
    clf = RandomizedSearchCV(aer, distributions, cv=fold_splits, scoring='accuracy', n_iter=N_ITER, random_state=random_state)
    search = clf.fit(X, y)
    return search.best_params_
