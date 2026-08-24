import pandas as pd, numpy as np, sys

def summarize(path, group_cols=("backbone","configuration")):
    df=pd.read_csv(path)
    metrics=[c for c in ["accuracy","macro_precision","macro_recall","macro_f1"] if c in df]
    out=df.groupby(list(group_cols))[metrics].agg(["mean","std"])
    print(out.to_string())

if __name__=="__main__":
    summarize(sys.argv[1])
