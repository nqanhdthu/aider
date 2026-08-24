import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

C_GRID=[1e-3,1e-2,1e-1,1,10,100]

def _build(kind, params, seed):
    if kind=="lr":
        return LogisticRegression(
            C=params["C"], class_weight=params["class_weight"],
            solver="lbfgs", tol=1e-4, max_iter=1000, penalty="l2"
        )
    if kind=="svm":
        return LinearSVC(
            C=params["C"], class_weight=params["class_weight"],
            loss="squared_hinge", penalty="l2", tol=1e-4, max_iter=1000
        )
    if kind=="knn":
        return KNeighborsClassifier(
            n_neighbors=params["k"], weights=params["weights"], metric="euclidean"
        )
    if kind=="dt":
        return DecisionTreeClassifier(
            criterion="gini", class_weight="balanced",
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed
        )
    if kind=="rf":
        return RandomForestClassifier(
            n_estimators=500, criterion="gini", max_features="sqrt",
            class_weight="balanced_subsample", max_depth=params["max_depth"],
            random_state=seed, n_jobs=-1
        )
    raise ValueError(kind)

def parameter_grid(kind):
    if kind in ("lr","svm"):
        return [{"C":c,"class_weight":w} for w in (None,"balanced") for c in C_GRID]
    if kind=="knn":
        return [{"k":k,"weights":w} for k in [1,3,5,7,9,11,15,21]
                for w in ("uniform","distance")]
    if kind=="dt":
        return [{"max_depth":d,"min_samples_leaf":m}
                for d in [10,20,40,None] for m in [1,5,10]]
    if kind=="rf":
        return [{"max_depth":d} for d in [20,40,None]]
    raise ValueError(kind)

def tune_training_only(X, y, kind, seed, n_splits=5):
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    best=None
    for params in parameter_grid(kind):
        scores=[]
        for tr,va in cv.split(X,y):
            scaler=StandardScaler().fit(X[tr])
            xtr=scaler.transform(X[tr]); xva=scaler.transform(X[va])
            model=_build(kind,params,seed)
            model.fit(xtr,y[tr])
            scores.append(f1_score(y[va],model.predict(xva),average="macro"))
        score=float(np.mean(scores))
        if best is None or score>best["cv_macro_f1"]:
            best={"params":params,"cv_macro_f1":score}
    scaler=StandardScaler().fit(X)
    model=_build(kind,best["params"],seed)
    model.fit(scaler.transform(X),y)
    return scaler,model,best
