"""Misura quanto il fine-tuning ha spostato il backbone rispetto ai pesi ImageNet.

Uso (dalla cartella di transfer_learning, con la .venv attiva):

  # 1) deriva dei pesi (non servono immagini)
  python diagnose_teacher.py --ckpt "D:\\...\\checkpoint_backbone\\resnet18_ce_15epochs_lr1e-4_batch16_seed42_splitSeed0.pth"

  # 2) in piu': deriva delle feature layer1-3 su immagini buone del train dell'anomaly detection
  python diagnose_teacher.py --ckpt "<stesso file>" --images "D:/emanuele/Code/dataset/tessuto_nero_dust_train/train/good" --n 64

  # 3) esporta i pesi ImageNet nello stesso formato, da passare a SK-RD4AD con --encoder-ckpt
  #    (controllo della pipeline: deve dare lo stesso risultato del run senza --encoder-ckpt)
  python diagnose_teacher.py --export-imagenet imagenet_resnet18.pth

Cosa stampa:
  - epoca / val_auroc salvati nel checkpoint (quale epoca e' stata scelta come "best")
  - variazione relativa ||W_ft - W_in|| / ||W_in|| per gruppo (conv1, layer1, layer2, layer3, layer4) e tipo
    (pesi conv, gamma/beta delle BN, statistiche running delle BN)
  - con --images: similarita' coseno media tra feature ImageNet e fine-tuned (per posizione) a layer1/2/3 e
    rapporto di partecipazione (rango efficace) delle feature: se cala molto, le feature si sono "appiattite"
    su poche direzioni e il decoder dello student le ricostruisce anche sulle anomalie.
"""
import argparse
import glob
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXTS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")


def imagenet_resnet18():
    # stessi pesi usati da SK-RD4AD (resnet18-f37072fd) e dal repo transfer_learning (ResNet18_Weights.DEFAULT)
    return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()


def load_finetuned(path):
    raw = torch.load(path, map_location="cpu")
    meta = {k: raw[k] for k in ("epoch", "val_auroc", "val_loss") if isinstance(raw, dict) and k in raw}
    sd = raw["encoder"] if isinstance(raw, dict) and "encoder" in raw else raw
    # parto dai pesi ImageNet: i tensori assenti dal checkpoint (fc, e layer4 se e' stato rimosso) restano ImageNet
    m = imagenet_resnet18()
    res = m.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith(("fc.", "layer4."))]
    if missing or res.unexpected_keys:
        raise RuntimeError(f"Chiavi non compatibili con resnet18. missing={missing[:5]} unexpected={res.unexpected_keys[:5]}")
    return m.eval(), meta


def weight_drift(m_in, m_ft):
    a, b = m_in.state_dict(), m_ft.state_dict()
    acc = {}
    for k in a:
        if k.startswith("fc.") or k.endswith("num_batches_tracked"):
            continue
        group = k.split(".")[0]
        if k.endswith(("running_mean", "running_var")):
            kind = "BN stats"
        elif a[k].ndim > 1:
            kind = "conv"
        else:
            kind = "BN affine"
        d = (b[k].double() - a[k].double()).pow(2).sum().item()
        n = a[k].double().pow(2).sum().item()
        s = acc.setdefault((group, kind), [0.0, 0.0])
        s[0] += d
        s[1] += n
    kinds = ["conv", "BN affine", "BN stats"]
    print("\nVariazione relativa dei pesi rispetto a ImageNet (%)")
    print(f"{'gruppo':<8}" + "".join(f"{k:>12}" for k in kinds))
    for g in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]:
        row = f"{g:<8}"
        for k in kinds:
            if (g, k) in acc:
                d, n = acc[(g, k)]
                row += f"{100 * (d ** 0.5) / (n ** 0.5 + 1e-12):>12.2f}"
            else:
                row += f"{'-':>12}"
        print(row)
    print("(layer4 non e' usato da SK-RD4AD: conta solo conv1, layer1, layer2, layer3)")


@torch.no_grad()
def features(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    f1 = m.layer1(x)
    f2 = m.layer2(f1)
    f3 = m.layer3(f2)
    return f1, f2, f3


def participation_ratio(f, max_pos=20000, seed=0):
    # f: (N, C, H, W) -> (N*H*W, C); PR = (sum lambda)^2 / sum lambda^2 sugli autovalori della covarianza
    x = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).double()
    if x.shape[0] > max_pos:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.shape[0], generator=g)[:max_pos]]
    x = x - x.mean(0, keepdim=True)
    ev = torch.linalg.eigvalsh(x.T @ x / (x.shape[0] - 1)).clamp_min(0)
    return (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30)).item()


def feature_drift(m_in, m_ft, folder, n, device):
    files = sorted(f for e in EXTS for f in glob.glob(os.path.join(folder, "**", f"*.{e}"), recursive=True))
    if not files:
        raise FileNotFoundError(f"Nessuna immagine in {folder}")
    files = random.Random(0).sample(files, min(n, len(files)))
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    m_in, m_ft = m_in.to(device), m_ft.to(device)

    cos = [0.0, 0.0, 0.0]
    fin, fft = [[], [], []], [[], [], []]
    seen = 0
    for i in range(0, len(files), 16):
        x = torch.stack([tf(Image.open(f).convert("RGB")) for f in files[i:i + 16]]).to(device)
        a, b = features(m_in, x), features(m_ft, x)
        for j in range(3):
            cos[j] += F.cosine_similarity(a[j], b[j], dim=1).mean().item() * x.shape[0]
            fin[j].append(a[j].cpu())
            fft[j].append(b[j].cpu())
        seen += x.shape[0]

    print(f"\nFeature su {seen} immagini ({folder})")
    print(f"{'layer':<8}{'cos(in,ft)':>12}{'PR ImageNet':>14}{'PR fine-tuned':>16}")
    for j, name in enumerate(["layer1", "layer2", "layer3"]):
        pin = participation_ratio(torch.cat(fin[j]))
        pft = participation_ratio(torch.cat(fft[j]))
        print(f"{name:<8}{cos[j] / seen:>12.4f}{pin:>14.2f}{pft:>16.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--images", type=str, default=None)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--export-imagenet", dest="export_imagenet", type=str, default=None)
    args = p.parse_args()

    m_in = imagenet_resnet18()

    if args.export_imagenet:
        sd = {k: v for k, v in m_in.state_dict().items() if not k.startswith("fc.")}
        torch.save({"encoder": sd}, args.export_imagenet)
        print(f"Salvato {args.export_imagenet} ({len(sd)} tensori, formato identico al best_encoder.pth)")
        return

    if not args.ckpt:
        p.error("serve --ckpt oppure --export-imagenet")

    m_ft, meta = load_finetuned(args.ckpt)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Metadati salvati: {meta if meta else 'nessuno'}  (epoch e' 0-based nel repo transfer_learning)")
    weight_drift(m_in, m_ft)

    if args.images:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feature_drift(m_in, m_ft, args.images, args.n, device)


if __name__ == "__main__":
    main()
