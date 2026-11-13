import csv
from PIL import Image
import pytest
from ml.src.data.dataset import CLASS_NAMES, FARZDataset, build_transforms

def test_dataset_loads_rgb_square_tensor(tmp_path):
    images=tmp_path/'images'; (images/'glass').mkdir(parents=True)
    Image.new('RGB',(512,384),(240,240,240)).save(images/'glass'/'one.jpg')
    manifest=tmp_path/'split.csv'
    with manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','class_name','split']); w.writeheader()
        w.writerow({'relative_path':'glass/one.jpg','class_name':'Glass','split':'train'})
    train_tf,_=build_transforms(224)
    x,y=FARZDataset(manifest,images,'train',train_tf)[0]
    assert tuple(x.shape)==(3,224,224) and y==0 and len(CLASS_NAMES)==5

def test_dataset_errors_when_images_are_missing(tmp_path):
    images=tmp_path/'images'; images.mkdir()
    manifest=tmp_path/'split.csv'
    with manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['relative_path','class_name','split']); w.writeheader()
        w.writerow({'relative_path':'paper/missing.jpg','class_name':'Paper','split':'train'})
    with pytest.raises(FileNotFoundError,match='images missing'):
        FARZDataset(manifest,images,'train')
