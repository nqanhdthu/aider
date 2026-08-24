import os, random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEEDS = list(range(202601, 202611))

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def inverse_frequency_weights(y, n_classes=None):
    y=np.asarray(y,dtype=int)
    if n_classes is None:
        n_classes=int(y.max())+1
    counts=np.bincount(y,minlength=n_classes)
    if np.any(counts==0):
        raise ValueError("Every class must occur in the training data.")
    return len(y)/(n_classes*counts.astype(float))

def classification_metrics(y_true, y_pred):
    p,r,f1,_=precision_recall_fscore_support(
        y_true,y_pred,average="macro",zero_division=0
    )
    return {
        "accuracy":100.0*accuracy_score(y_true,y_pred),
        "macro_precision":100.0*p,
        "macro_recall":100.0*r,
        "macro_f1":100.0*f1,
    }
