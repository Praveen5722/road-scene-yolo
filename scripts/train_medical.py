"""Train the regularized CNN using an ImageFolder dataset."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from medical_cnn.model import RegularizedCNN

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/medical_images"))
    p.add_argument("--output", type=Path, default=Path("artifacts/medical_cnn.pt"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=5)
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1: p.error("epochs and batch-size must be positive")
    train_dir, val_dir = a.data / "train", a.data / "val"
    if not train_dir.is_dir() or not val_dir.is_dir(): p.error(f"Expected train/ and val/ under {a.data}")
    aug = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(8), transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    plain = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    tr, va = datasets.ImageFolder(train_dir, transform=aug), datasets.ImageFolder(val_dir, transform=plain)
    if tr.classes != va.classes: p.error("Train and validation class folders must match")
    if not len(tr) or not len(va): p.error("Dataset folders must contain images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=a.batch_size, shuffle=False, num_workers=0)
    model = RegularizedCNN(len(tr.classes), a.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    best, stale, history = float("inf"), 0, []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.epochs + 1):
        model.train(); train_sum = correct = total = 0
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step()
            train_sum += loss.item()*len(y); correct += (logits.argmax(1)==y).sum().item(); total += len(y)
        model.eval(); val_sum = val_correct = val_total = 0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(device),y.to(device); logits=model(x); loss=loss_fn(logits,y)
                val_sum += loss.item()*len(y); val_correct += (logits.argmax(1)==y).sum().item(); val_total += len(y)
        row={"epoch":epoch,"train_loss":train_sum/total,"train_accuracy":correct/total,"val_loss":val_sum/val_total,"val_accuracy":val_correct/val_total}
        history.append(row); print(json.dumps(row))
        if row["val_loss"] < best:
            best, stale = row["val_loss"], 0
            torch.save({"model_state_dict":model.state_dict(),"classes":tr.classes,"image_size":224,"dropout":a.dropout,"validation_accuracy":row["val_accuracy"]},a.output)
        else:
            stale += 1
            if stale >= a.patience: print("Early stopping."); break
    a.output.with_suffix(".history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    print(f"Saved best checkpoint to {a.output}; device={device}")

if __name__ == "__main__": main()
