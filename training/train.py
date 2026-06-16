"""
Usage:
  python training/train.py --model transformer
  python training/train.py --model crnn
  python training/train.py --model all
"""

import argparse, os, json, time, random, warnings, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, accuracy_score,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.dataset import ClipDataset, CLASSES, NUM_CLASSES
from models.transformer import TransformerModel
from models.crnn import CRNNModel

warnings.filterwarnings("ignore")

dr        = "dataset"
sd        = "checkpoints"
mxc       = 70
ep        = 5
bs        = 4
lr0       = 3e-4
wd0       = 1e-4
SEED      = 42
ALL_MODELS = ["crnn", "transformer"]

ml = {
    "crnn"       : "CRNN",
    "transformer": "Transformer",
}

mr = {
    "crnn"       : CRNNModel,
    "transformer": TransformerModel,
}

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(mc):
    td = ClipDataset(dr, split="train", max_clips=mc)
    vd = ClipDataset(dr, split="val",   max_clips=mc)

    sw  = td.class_weights()
    smp = WeightedRandomSampler(sw, len(sw), replacement=True)

    tdl = DataLoader(td, batch_size=bs, sampler=smp, num_workers=4,
                     pin_memory=True, persistent_workers=True)
    vdl = DataLoader(vd, batch_size=bs, shuffle=False, num_workers=4,
                     pin_memory=True, persistent_workers=True)

    cnt = defaultdict(int)
    for _, lb in td.samples:
        cnt[CLASSES[lb]] += 1
    print(f"  train clips: {dict(cnt)}  |  val clips: {len(vd)}")
    return tdl, vdl, td, vd


def compute_metrics(labels, preds):
    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall"   : recall_score(labels, preds, average="macro", zero_division=0),
    }


def train_epoch(mdl, ldr, opt, crit, scl):
    mdl.train()
    tl = cr = n = 0
    for xb, yb in ldr:
        xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEV.type == "cuda")):
            lg  = mdl(xb)
            lss = crit(lg, yb)
        scl.scale(lss).backward()
        scl.unscale_(opt)
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        scl.step(opt)
        scl.update()
        tl += lss.item() * len(yb)
        cr += (lg.argmax(1) == yb).sum().item()
        n  += len(yb)
    return tl / max(n, 1), cr / max(n, 1)


@torch.no_grad()
def eval_epoch(mdl, ldr, crit):
    mdl.eval()
    tl = cr = n = 0
    ap, al = [], []
    for xb, yb in ldr:
        xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEV.type == "cuda")):
            lg  = mdl(xb)
            lss = crit(lg, yb)
        tl += lss.item() * len(yb)
        pr  = lg.argmax(1)
        cr += (pr == yb).sum().item()
        n  += len(yb)
        ap += pr.cpu().tolist()
        al += yb.cpu().tolist()
    return tl / max(n, 1), cr / max(n, 1), ap, al


def run_neural(mn, tdl, vdl, epochs) -> dict:
    print(f"\n{'═'*62}")
    print(f"  Training: {ml[mn]}")
    print(f"{'═'*62}")

    mdl  = mr[mn]().to(DEV)
    tp   = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  Parameters : {tp:,}")

    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt   = optim.AdamW(mdl.parameters(), lr=lr0, weight_decay=wd0)
    sch   = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr0,
        steps_per_epoch=len(tdl), epochs=epochs, pct_start=0.2,
    )
    scl   = torch.amp.GradScaler("cuda", enabled=(DEV.type == "cuda"))

    hist     = defaultdict(list)
    bva      = -1.0
    cp       = os.path.join(sd, f"{mn}_best.pt")

    for ep2 in range(1, epochs + 1):
        t0 = time.time()
        trl, tra = train_epoch(mdl, tdl, opt, crit, scl)
        val, vaa, preds, labels = eval_epoch(mdl, vdl, crit)
        sch.step()
        el = time.time() - t0
        m  = compute_metrics(labels, preds)

        hist["tr_loss"].append(trl)
        hist["tr_acc"].append(tra)
        hist["va_loss"].append(val)
        hist["va_acc"].append(vaa)
        hist["va_f1"].append(m["f1"])
        hist["va_precision"].append(m["precision"])
        hist["va_recall"].append(m["recall"])

        fg = ""
        if vaa > bva:
            bva = vaa
            torch.save(mdl.state_dict(), cp)
            fg = "  ✓ saved"

        print(f"  Epoch {ep2:>2}/{epochs} | "
              f"tr_loss={trl:.4f}  tr_acc={tra:.3f} | "
              f"va_loss={val:.4f}  va_acc={vaa:.3f}  "
              f"f1={m['f1']:.3f}  prec={m['precision']:.3f}  "
              f"rec={m['recall']:.3f} | {el:.1f}s{fg}")

    mdl.load_state_dict(torch.load(cp, map_location=DEV, weights_only=True))
    _, _, fp2, fl2 = eval_epoch(mdl, vdl, crit)
    fm  = compute_metrics(fl2, fp2)
    ba  = balanced_accuracy_score(fl2, fp2)

    print(f"\n  Best val acc : {bva:.4f}")
    print(f"  Accuracy     : {fm['accuracy']:.4f}")
    print(f"  Macro F1     : {fm['f1']:.4f}")
    print(f"  Balanced acc : {ba:.4f}")
    print(classification_report(fl2, fp2, target_names=CLASSES, zero_division=0))

    return {
        "model"    : mn,
        "best_acc" : bva,
        "accuracy" : fm["accuracy"],
        "macro_f1" : fm["f1"],
        "precision": fm["precision"],
        "recall"   : fm["recall"],
        "bal_acc"  : ba,
        "history"  : dict(hist),
        "preds"    : fp2,
        "labels"   : fl2,
        "cm"       : confusion_matrix(fl2, fp2).tolist(),
        "params"   : tp,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="transformer", choices=ALL_MODELS + ["all"])
    p.add_argument("--epochs",     type=int,   default=ep)
    p.add_argument("--batch_size", type=int,   default=bs)
    p.add_argument("--lr",         type=float, default=lr0)
    p.add_argument("--data_root",  type=str,   default=dr)
    p.add_argument("--save_dir",   type=str,   default=sd)
    p.add_argument("--max_clips",  type=int,   default=mxc)
    return p.parse_args()


def main():
    args = parse_args()

    global dr, sd, ep, bs, lr0, mxc
    dr  = args.data_root
    sd  = args.save_dir
    ep  = args.epochs
    bs  = args.batch_size
    lr0 = args.lr
    mxc = args.max_clips

    os.makedirs(sd, exist_ok=True)
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    if DEV.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    print(f"\n{'═'*62}")
    print("Driving Behavior Classification")
    print(f"{'═'*62}")
    print(f"  Device    : {DEV}"
          + (f"  ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else ""))
    print(f"  Data root : {dr}")
    print(f"  Save dir  : {sd}")
    print(f"  Model(s)  : {args.model}")
    print(f"  Epochs    : {ep}   Batch: {bs}   LR: {lr0}")

    tr = ALL_MODELS if args.model == "all" else [args.model]
    tdl, vdl, td, vd = make_loaders(mxc)

    ar = []
    for mn in tr:
        r = run_neural(mn, tdl, vdl, ep)
        ar.append(r)
        jp = os.path.join(sd, f"{mn}_results.json")
        with open(jp, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  [json] → {jp}")

    if len(ar) > 1:
        print(f"\n[summary]")
        print(f"  {'Model':<15} {'Accuracy':>9} {'Macro F1':>9} {'Bal Acc':>9}")
        print(f"  {'-'*45}")
        for r in ar:
            nm = ml.get(r["model"], r["model"])
            print(f"  {nm:<15} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f} {r['bal_acc']:>9.4f}")

    print(f"\n[done] results saved to: {sd}/\n")


if __name__ == "__main__":
    main()
