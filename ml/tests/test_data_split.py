from collections import defaultdict
from ml.src.data.create_data_split import create_split, build_group_ids

def synthetic():
    rows=[]; pairs=[]
    for cls in ["Glass","Paper","Cardboard","Plastic","Metal"]:
        key=cls.lower()
        for i in range(14):
            rows.append({"relative_path":f"{key}/{key}_{i}.jpg","class_name":cls,"sha256":f"{key}_{i}"})
        pairs.append({"phash_hamming_distance":"2","pixel_identical":"False",
          "class_a":key,"file_a":f"{key}_0.jpg","class_b":key,"file_b":f"{key}_1.jpg"})
    return rows,pairs

def test_grouped_split_has_no_leakage():
    rows,pairs=synthetic(); result=create_split(rows,pairs,seed=42)
    group_splits=defaultdict(set)
    for row in result: group_splits[row["group_id"]].add(row["split"])
    assert len(result)==70
    assert all(len(value)==1 for value in group_splits.values())
    assert {row["split"] for row in result}=={"train","validation","test"}
    assert len(group_splits)==65

def test_pixel_identical_pair_is_not_grouped_as_perceptual_candidate():
    rows,_=synthetic()
    pair=[{"phash_hamming_distance":"0","pixel_identical":"True","class_a":"glass",
      "file_a":"glass_0.jpg","class_b":"glass","file_b":"glass_1.jpg"}]
    groups=build_group_ids(rows,pair)
    assert groups["glass/glass_0.jpg"] != groups["glass/glass_1.jpg"]
