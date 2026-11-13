"""Train MobileNetV3-Small. Validation selects the checkpoint; test stays untouched."""
from __future__ import annotations
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ml.src.data.dataset import CLASS_NAMES,FARZDataset,IMAGENET_MEAN,IMAGENET_STD,build_transforms
from ml.src.models.mobilenet import build_model

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def run_epoch(model,loader,criterion,device,optimizer=None,scaler=None,amp=False,desc=""):
    training=optimizer is not None;model.train(training)
    total_loss,y_true,y_pred=0.0,[],[]
    for images,targets in tqdm(loader,desc=desc,leave=False):
        images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
        if training:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                logits=model(images);loss=criterion(logits,targets)
            if training:
                if amp and scaler is not None:
                    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                else:loss.backward();optimizer.step()
        total_loss+=loss.detach().item()*targets.size(0)
        y_true.extend(targets.detach().cpu().tolist());y_pred.extend(logits.detach().argmax(1).cpu().tolist())
    return {"loss":total_loss/len(loader.dataset),"accuracy":float(accuracy_score(y_true,y_pred)),
            "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0))}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--manifest",type=Path,default=None);p.add_argument("--images-dir",type=Path,default=None)
    p.add_argument("--output-dir",type=Path,default=None);p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--patience",type=int,default=5);p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--no-pretrained",action="store_true")
    p.add_argument("--no-amp",action="store_true");return p.parse_args()

def main():
    args=parse_args();root=args.root.resolve()
    manifest=(args.manifest or root/"data/manifests/data_split.csv").resolve()
    images_dir=(args.images_dir or root/"data/raw/dataset-resized").resolve()
    output=(args.output_dir or root/"ml/outputs/mobilenet_v3_small").resolve();output.mkdir(parents=True,exist_ok=True)
    if min(args.epochs,args.batch_size,args.patience)<1 or args.num_workers<0:
        raise ValueError("epochs, batch-size, patience must be positive; num-workers cannot be negative")
    set_seed(args.seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp=device.type=="cuda" and not args.no_amp
    print(f"Device={device} | AMP={amp} | pretrained={not args.no_pretrained}")
    train_tf,eval_tf=build_transforms(args.image_size)
    train_ds=FARZDataset(manifest,images_dir,"train",train_tf)
    val_ds=FARZDataset(manifest,images_dir,"validation",eval_tf)
    common=dict(batch_size=args.batch_size,num_workers=args.num_workers,pin_memory=device.type=="cuda",
                persistent_workers=args.num_workers>0)
    train_loader=DataLoader(train_ds,shuffle=True,**common)
    val_loader=DataLoader(val_ds,shuffle=False,**common)
    model=build_model(pretrained=not args.no_pretrained).to(device)
    criterion=nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer=AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    scheduler=ReduceLROnPlateau(optimizer,mode="max",factor=0.3,patience=2)
    try:scaler=torch.amp.GradScaler("cuda",enabled=amp)
    except (AttributeError,TypeError):scaler=torch.cuda.amp.GradScaler(enabled=amp)
    best_f1,best_epoch,stale=-1.0,0,0;history=[]
    for epoch in range(1,args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics=run_epoch(model,train_loader,criterion,device,optimizer,scaler,amp,"train")
        val_metrics=run_epoch(model,val_loader,criterion,device,desc="validation")
        lr=float(optimizer.param_groups[0]["lr"]);scheduler.step(val_metrics["macro_f1"])
        row={"epoch":epoch,"learning_rate":lr,
             **{f"train_{k}":v for k,v in train_metrics.items()},
             **{f"val_{k}":v for k,v in val_metrics.items()}}
        history.append(row)
        with (output/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print(json.dumps(row,indent=2))
        if val_metrics["macro_f1"]>best_f1:
            best_f1,best_epoch,stale=val_metrics["macro_f1"],epoch,0
            checkpoint={"architecture":"mobilenet_v3_small","num_classes":len(CLASS_NAMES),
              "class_names":list(CLASS_NAMES),"class_to_idx":{n:i for i,n in enumerate(CLASS_NAMES)},
              "image_size":args.image_size,"mean":list(IMAGENET_MEAN),"std":list(IMAGENET_STD),
              "epoch":epoch,"validation_metrics":val_metrics,"model_state_dict":model.state_dict(),"seed":args.seed}
            torch.save(checkpoint,output/"best_model.pth");print(f"Saved best checkpoint: val macro-F1={best_f1:.4f}")
        else:stale+=1
        if stale>=args.patience:print("Early stopping.");break
    summary={"architecture":"mobilenet_v3_small","best_epoch":best_epoch,"best_validation_macro_f1":best_f1,
      "epochs_completed":len(history),"device":str(device),"pretrained_imagenet_weights":not args.no_pretrained,
      "test_split_used_during_training":False,"checkpoint":str(output/"best_model.pth")}
    (output/"training_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Training finished; test split remained untouched.");print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
