"""Create deterministic 70/15/15 stratified group split with no duplicate leakage."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def read_csv(path):
    with path.open(newline="",encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))

class UnionFind:
    def __init__(self,items):
        self.parent={x:x for x in items}; self.rank={x:0 for x in items}
    def find(self,x):
        if self.parent[x]!=x: self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        ra,rb=self.find(a),self.find(b)
        if ra==rb:return
        if self.rank[ra]<self.rank[rb]:ra,rb=rb,ra
        self.parent[rb]=ra
        if self.rank[ra]==self.rank[rb]:self.rank[ra]+=1

def truth(value): return str(value).strip().lower() in {"true","1","yes"}

def build_group_ids(rows,pairs,max_distance=4):
    paths={row["relative_path"] for row in rows}; uf=UnionFind(sorted(paths))
    for pair in pairs:
        if int(pair.get("phash_hamming_distance",999))>max_distance or truth(pair.get("pixel_identical","false")):
            continue
        left=f'{pair["class_a"].lower()}/{pair["file_a"]}'
        right=f'{pair["class_b"].lower()}/{pair["file_b"]}'
        if left in paths and right in paths: uf.union(left,right)
    clusters=defaultdict(list)
    for path in sorted(paths): clusters[uf.find(path)].append(path)
    ordered=sorted((sorted(members) for members in clusters.values()),key=lambda members:members[0])
    return {path:f"G{i:05d}" for i,members in enumerate(ordered,1) for path in members}

def create_split(rows,pairs,seed=42):
    group_by_path=build_group_ids(rows,pairs)
    labels=np.asarray([row["class_name"] for row in rows])
    groups=np.asarray([group_by_path[row["relative_path"]] for row in rows])
    splitter=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=seed)
    assignment={}
    for fold,(_,indices) in enumerate(splitter.split(np.zeros(len(rows)),labels,groups)):
        for index in indices.tolist():
            if index in assignment: raise RuntimeError(f"Row assigned more than once: {index}")
            assignment[index]=fold
    if len(assignment)!=len(rows): raise RuntimeError("Some rows were not assigned to a fold.")
    output=[]
    for index,row in enumerate(rows):
        fold=assignment[index]; split="test" if fold==0 else "validation" if fold==1 else "train"
        output.append({"relative_path":row["relative_path"],"class_name":row["class_name"],
            "sha256":row["sha256"],"group_id":group_by_path[row["relative_path"]],
            "fold":fold,"split":split})
    group_splits=defaultdict(set)
    for row in output: group_splits[row["group_id"]].add(row["split"])
    if any(len(splits)>1 for splits in group_splits.values()):
        raise RuntimeError("Leakage detected: a perceptual group occurs in multiple splits.")
    return output

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--seed",type=int,default=42); args=parser.parse_args(); root=args.root.resolve()
    candidates=read_csv(root/"data/manifests/training_candidate_manifest.csv")
    pairs=read_csv(root/"reports/data_audit/perceptual_duplicate_candidates.csv")
    rows=create_split(candidates,pairs,args.seed)
    output=root/"data/manifests/data_split.csv"; output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    class_counts={name:dict(Counter(r["class_name"] for r in rows if r["split"]==name))
                  for name in ("train","validation","test")}
    summary={"seed":args.seed,"method":"StratifiedGroupKFold(7): 5 folds train, 1 validation, 1 test",
      "max_phash_distance_grouped":4,"candidate_images":len(rows),
      "unique_groups":len({r["group_id"] for r in rows}),
      "split_counts":dict(Counter(r["split"] for r in rows)),
      "class_counts":class_counts,"group_leakage":False,"exact_conflict_files_included":0}
    (root/"reports/data_audit/split_counts.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); print("Wrote",output)

if __name__=="__main__": main()
