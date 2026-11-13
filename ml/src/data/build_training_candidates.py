"""Hold out every member of byte-identical cross-label pairs pending human review."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path

def read(path):
    with path.open(newline="",encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path.cwd())
    args=parser.parse_args(); root=args.root.resolve()
    manifest=read(root/"data/manifests/five_class_manifest.csv")
    duplicates=read(root/"reports/data_audit/exact_duplicates.csv")
    held_paths={r["relative_path"] for r in duplicates if r["duplicate_type"]=="cross_label_conflict"}
    held=[r for r in manifest if r["relative_path"] in held_paths]
    candidates=[r for r in manifest if r["relative_path"] not in held_paths]
    destination=root/"data/manifests/training_candidate_manifest.csv"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(candidates[0])); writer.writeheader(); writer.writerows(candidates)
    review_path=root/"reports/data_audit/held_out_conflicting_duplicates.csv"
    with review_path.open("w",newline="",encoding="utf-8") as file:
        fields=["class_name","relative_path","sha256","reason"]; writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        for row in held:
            writer.writerow({"class_name":row["class_name"],"relative_path":row["relative_path"],
              "sha256":row["sha256"],"reason":"byte-identical content appears under conflicting labels; pending manual review"})
    summary={"full_manifest_images":len(manifest),"training_candidate_images":len(candidates),
        "held_out_images":len(held),"training_candidate_class_counts":dict(Counter(r["class_name"] for r in candidates)),
        "physical_source_images_deleted_or_modified":0,"split_created":False}
    (root/"reports/data_audit/training_candidate_counts.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
