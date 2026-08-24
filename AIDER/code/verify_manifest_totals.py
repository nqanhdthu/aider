from pathlib import Path
import csv, sys

EXPECTED={
    "ip102_train.txt":45095,
    "ip102_val.txt":7508,
    "ip102_test.txt":22619,
}

def count_txt(path):
    return sum(1 for x in open(path,encoding="utf-8") if x.strip() and not x.startswith("#"))

def count_csv(path):
    with open(path,newline="",encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))

def main(root):
    root=Path(root)
    ok=True
    for fn,n in EXPECTED.items():
        p=root/fn
        if not p.exists():
            print("MISSING",p); ok=False; continue
        got=count_txt(p)
        print(fn,got,"expected",n)
        ok &= got==n
    for fn,n in [("xie24_split_manifest.csv",1600),("d0_split_manifest.csv",4508)]:
        p=root/fn
        if not p.exists():
            print("MISSING",p); ok=False; continue
        got=count_csv(p)
        print(fn,got,"expected",n)
        ok &= got==n
    raise SystemExit(0 if ok else 1)

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else ".")
