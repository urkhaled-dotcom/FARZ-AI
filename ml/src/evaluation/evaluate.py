"""Evaluate the best checkpoint once on the untouched test split."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score,classification_report,ConfusionMatrixDisplay,confusion_matrix,f1_score,precision_score,recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from ml.src.data.dataset import CLASS_NAMES,FARZDataset,build_transforms
from ml.src.models.mobilenet import build_model

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--checkpoint",type=Path,default=None);p.add_argument("--manifest",type=Path,default=None)
    p.add_argument("--images-dir",type=Path,default=None);p.add_argument("--output-dir",type=Path,default=None)
    p.add_argument("--batch-size",type=int,default=32);p.add_argument("--num-workers",type=int,default=2)
    args=p.parse_args();root=args.root.resolve()
    ckpt=(args.checkpoint or root/"ml/outputs/mobilenet_v3_small/best_model.pth").resolve()
    manifest=(args.manifest or root/"data/manifests/data_split.csv").resolve()
    images=(args.images_dir or root/"data/raw/dataset-resized").resolve()
    out=(args.output_dir or root/"ml/outputs/mobilenet_v3_small/test_evaluation").resolve();out.mkdir(parents=True,exist_ok=True)
    if not ckpt.is_file():raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    checkpoint=torch.load(ckpt,map_location="cpu",weights_only=False)
    classes=checkpoint.get("class_names",list(CLASS_NAMES))
    if tuple(classes)!=tuple(CLASS_NAMES):raise ValueError(f"Checkpoint class order mismatch: {classes}")
    _,transform=build_transforms(int(checkpoint.get("image_size",224)))
    dataset=FARZDataset(manifest,images,"test",transform)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
      pin_memory=torch.cuda.is_available(),persistent_workers=args.num_workers>0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(pretrained=False).to(device);model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
    y_true=[];y_pred=[];records=[];total_loss=0.0;criterion=torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for batch,targets in tqdm(loader,desc="test"):
            batch=batch.to(device);targets=targets.to(device);logits=model(batch);loss=criterion(logits,targets)
            probs=logits.softmax(dim=1);confidence,prediction=probs.max(dim=1)
            total_loss+=float(loss.item())*targets.size(0)
            y_true.extend(targets.cpu().tolist());y_pred.extend(prediction.cpu().tolist())
            for truth,pred,score in zip(targets.cpu().tolist(),prediction.cpu().tolist(),confidence.cpu().tolist()):
                records.append({"true_class":classes[truth],"predicted_class":classes[pred],
                                "top1_score":score,"correct":truth==pred})
    report=classification_report(y_true,y_pred,labels=list(range(len(classes))),target_names=classes,
      output_dict=True,zero_division=0)
    metrics={"test_images":len(dataset),"test_loss":total_loss/len(dataset),
      "accuracy":float(accuracy_score(y_true,y_pred)),
      "macro_precision":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
      "macro_recall":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
      "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
      "class_names":classes,"checkpoint_epoch":checkpoint.get("epoch"),"device":str(device),
      "note":"Softmax scores are not calibrated probabilities."}
    (out/"test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (out/"classification_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (out/"test_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    matrix=confusion_matrix(y_true,y_pred,labels=list(range(len(classes))))
    fig,ax=plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay(matrix,display_labels=classes).plot(ax=ax,cmap="Greens",colorbar=False,values_format="d")
    ax.set_title("FARZ — Test Confusion Matrix");fig.tight_layout()
    fig.savefig(out/"confusion_matrix.png",dpi=160);plt.close(fig)
    print(json.dumps(metrics,indent=2));print("Results saved to:",out)

if __name__=="__main__":main()
