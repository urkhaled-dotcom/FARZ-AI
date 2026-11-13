"""Build a five-class image manifest from an extracted TrashNet archive."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

CLASSES=("glass","paper","cardboard","plastic","metal")
EXTENSIONS={".jpg",".jpeg",".png",".webp",".bmp"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir",type=Path,default=Path("data/raw/dataset-resized"))
    parser.add_argument("--root",type=Path,default=Path.cwd())
    args=parser.parse_args()
    root=args.root.resolve(); images=args.images_dir.resolve()
    if not images.is_dir():
        raise FileNotFoundError(f"Expected extracted dataset directory: {images}")
    rows=[]; hash_groups=defaultdict(list); invalid=[]
    for class_name in CLASSES:
        folder=images/class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS or path.name.startswith("._"):
                continue
            payload=path.read_bytes(); digest=hashlib.sha256(payload).hexdigest()
            width=height=None; mode="unknown"; valid=True
            try:
                with Image.open(path) as image: image.verify()
                with Image.open(path) as image: width,height,mode=image.width,image.height,image.mode
            except Exception:
                valid=False; invalid.append(str(path))
            row={"relative_path":f"{class_name}/{path.name}","class_name":class_name.title(),
                 "file_name":path.name,"sha256":digest,"bytes":len(payload),
                 "width":width,"height":height,"mode":mode,"valid_image":valid}
            rows.append(row); hash_groups[digest].append(row)
    if not rows:
        raise RuntimeError("No images found in the five approved class folders.")
    manifest=root/"data/manifests/five_class_manifest.csv"; manifest.parent.mkdir(parents=True,exist_ok=True)
    with manifest.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    duplicate_rows=[]
    for digest,members in hash_groups.items():
        if len(members)<2: continue
        labels=sorted({r["class_name"] for r in members})
        duplicate_type="cross_label_conflict" if len(labels)>1 else "same_class_duplicate"
        for row in members:
            duplicate_rows.append({"sha256":digest,"duplicate_type":duplicate_type,
                "class_name":row["class_name"],"relative_path":row["relative_path"],
                "all_labels_in_group":"|".join(labels)})
    duplicate_path=root/"reports/data_audit/exact_duplicates.csv"
    duplicate_path.parent.mkdir(parents=True,exist_ok=True)
    with duplicate_path.open("w",newline="",encoding="utf-8") as file:
        columns=["sha256","duplicate_type","class_name","relative_path","all_labels_in_group"]
        writer=csv.DictWriter(file,fieldnames=columns); writer.writeheader(); writer.writerows(duplicate_rows)
    report={"class_counts":dict(Counter(r["class_name"] for r in rows)),
        "total_five_class_images":len(rows),"invalid_images":invalid,
        "exact_duplicate_groups":sum(len(g)>1 for g in hash_groups.values()),
        "cross_label_duplicate_groups":sum(len(g)>1 and len({r["class_name"] for r in g})>1 for g in hash_groups.values()),
        "source_was_modified":False}
    (root/"reports/data_audit/class_counts.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if invalid: raise SystemExit("Invalid images found; review before training.")

if __name__=="__main__": main()
