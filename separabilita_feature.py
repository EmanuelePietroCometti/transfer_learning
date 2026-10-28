#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test di separabilità diretta delle feature: ImageNet contro i Teacher adattati.

DISEGNO (fissato prima di vedere i risultati)
---------------------------------------------
Domanda.  Il layer3 adattato separa le difettose dalla polvere meglio del layer3 ImageNet?
Senza addestrare nessuno Student: solo forward pass della ResNet-18.

Encoder.  ImageNet e i Teacher passati con --teacher. Conv1, bn1, layer1 e layer2 dei Teacher sono
          identici all'ImageNet: lo script lo verifica sui pesi e, se è vero, riusa i punteggi
          dell'ImageNet per quei layer (layer1 e layer2 sono quindi il controllo, non un confronto).
Dati.     Per ogni versione (V1, V2): banca = tutte le patch di train/good (polvere inclusa dove c'è),
          test = test/good + test/<difetti>. Le immagini di polvere sono quelle della lista --dust.
Score.    Stile PatchCore, per layer: feature del layer, media locale 3x3, distanza euclidea dal vicino
          più prossimo nella banca (ricerca esatta, niente coreset); score dell'immagine = massimo sulle
          patch. Facoltativo (--mode global): kNN con k=1 su un vettore per immagine (media + massimo
          spaziali), come la baseline kNN dell'EDA.
Metriche. I-AUROC con positivi = difettose e negativi =
            DvGP: good + polvere    (metrica primaria, la stessa della Tab. 5.5)
            DvP : solo polvere      (la domanda del test)
            DvG : solo good puliti  (controllo: separabilità senza polvere)
Bootstrap. 2000 ricampionamenti delle immagini, stratificati per gruppo (difettose, good, polvere),
          con gli STESSI indici per tutti gli encoder (bootstrap appaiato). Intervallo al 95%
          (percentili 2,5 e 97,5) della differenza  AUROC(Teacher) - AUROC(ImageNet).
Regola.   Un Teacher è utile su una versione solo se l'estremo inferiore dell'intervallo della
          differenza, sul layer3, è > 0 sia per DvGP sia per DvP. Altrimenti NO.
          Con 41 immagini di polvere l'intervallo è largo: è voluto, e fa parte del risultato.

USO
---
python separabilita_feature.py --data-root /path/dataset \
    --dust V1=/path/polvere_v1.txt --dust V2=/path/polvere_v2.txt \
    --teacher lam0=/path/resnet18_l3head_b1_lamPreserve0.pth \
    --teacher lam5=/path/resnet18_l3head_b1_lamPreserve5.pth \
    --teacher lam20=/path/resnet18_l3head_b1_lamPreserve20.pth \
    --out risultati_separabilita

--dust: file di testo con un nome di file per riga (anche un percorso: conta il nome), oppure una cartella
        che contiene le immagini di polvere. Senza "V=" vale per tutte le versioni.
"""
import argparse
import copy
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
METRICS = [("DvGP", "difettose contro good e polvere"),
           ("DvP", "difettose contro polvere"),
           ("DvG", "difettose contro good puliti")]
DEFAULT_VERSIONS = {"V1": "tessuto_nero_dust_validation", "V2": "tessuto_nero_dust_train"}


# ----------------------------------------------------------------------------- dati
def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXT)


def load_dust_names(spec):
    p = Path(spec)
    if p.is_dir():
        return {q.name for q in list_images(p)}
    names = set()
    for line in p.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.add(Path(line.replace("\\", "/")).name)
    return names


def parse_kv(items, default_key=None):
    out = {}
    for it in items or []:
        if "=" in it and not Path(it).exists():
            k, v = it.split("=", 1)
            out[k] = v
        elif default_key is not None:
            out[default_key] = it
        else:
            sys.exit(f"Argomento '{it}': usa la forma NOME=PERCORSO")
    return out


def batches(paths, tfm, bs, device):
    for i in range(0, len(paths), bs):
        x = torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
        yield x.to(device)


# ----------------------------------------------------------------------------- modelli
BACKBONE_KEY = re.compile(r"(conv1|bn1|layer[1-4])\.")


# Se l'encoder è un nn.Sequential(*resnet.children()[:-2]) le chiavi sono numeriche: 0 conv1, 1 bn1, 4-7 layer1-4.
SEQ_NAMES = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def _tensor_dicts(obj, depth=0):
    """Tutti i dict che contengono tensori, anche annidati (profondità massima 4)."""
    if isinstance(obj, torch.nn.Module):
        yield obj.state_dict()
        return
    if isinstance(obj, dict):
        if any(torch.is_tensor(v) for v in obj.values()):
            yield obj
        if depth < 4:
            for v in obj.values():
                if isinstance(v, (dict, torch.nn.Module)):
                    yield from _tensor_dicts(v, depth + 1)


def _has_layers(d):
    return all(any(k.startswith(f"layer{i}.") for k in d) for i in (1, 2, 3))


def _backbone_from(sd):
    a = {}
    for k, v in sd.items():
        m = BACKBONE_KEY.search(k)
        if torch.is_tensor(v) and m:
            a[k[m.start():]] = v
    if _has_layers(a):
        return a
    b = {}  # variante Sequential con chiavi numeriche
    for k, v in sd.items():
        mm = re.search(r"(?:^|\.)(\d)\.(.+)$", k)
        if torch.is_tensor(v) and mm and mm.group(1) in SEQ_NAMES:
            b[f"{SEQ_NAMES[mm.group(1)]}.{mm.group(2)}"] = v
    return b if _has_layers(b) else a


def extract_backbone_state(obj):
    """Estrae dal checkpoint i soli pesi della ResNet-18 (conv1, bn1, layer1-4), togliendo i prefissi.
    Cerca anche nei dict annidati e accetta un nn.Module intero o un encoder nn.Sequential."""
    best = {}
    for sd in _tensor_dicts(obj):
        cand = _backbone_from(sd)
        if len(cand) > len(best):
            best = cand
    return best


def describe(obj, indent=0, depth=0):
    """Descrizione breve della struttura di un checkpoint, per capire perché non si riconosce."""
    pad = " " * indent
    if isinstance(obj, torch.nn.Module):
        print(f"{pad}nn.Module {type(obj).__name__}: prime chiavi {list(obj.state_dict())[:8]}")
    elif isinstance(obj, dict):
        print(f"{pad}dict con {len(obj)} chiavi")
        for i, (k, v) in enumerate(obj.items()):
            if i >= 12:
                print(f"{pad}  ...")
                break
            if torch.is_tensor(v):
                print(f"{pad}  {k!r}: tensor {tuple(v.shape)}")
            elif isinstance(v, (dict, torch.nn.Module)) and depth < 2:
                print(f"{pad}  {k!r}:")
                describe(v, indent + 4, depth + 1)
            else:
                print(f"{pad}  {k!r}: {type(v).__name__}")
    else:
        print(f"{pad}{type(obj).__name__}")


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_model(state=None, imagenet=False, base=None):
    """Se si passa base, parte da una sua copia: i pesi assenti nel checkpoint (layer4, non usato) restano quelli di base."""
    if base is not None:
        m = copy.deepcopy(base).cpu()
    elif imagenet:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = torchvision.models.resnet18(weights=None)
    if state is not None:
        res = m.load_state_dict(state, strict=False)
        miss = [k for k in res.missing_keys if not k.startswith(("fc.", "layer4.")) and "num_batches_tracked" not in k]
        if miss:
            sys.exit(f"Chiavi del backbone mancanti nel checkpoint: {miss[:6]} ...")
    return m.eval()


@torch.no_grad()
def extract(model, x, layers):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    f1 = model.layer1(x)
    f2 = model.layer2(f1)
    f3 = model.layer3(f2)
    allf = {1: f1, 2: f2, 3: f3}
    return {l: allf[l] for l in layers}


def same_upto(sd_a, sd_b, layer):
    pref = ("conv1.", "bn1.") + tuple(f"layer{i}." for i in range(1, layer + 1))
    for k, v in sd_a.items():
        if k.startswith(pref) and (k not in sd_b or not torch.equal(v.cpu(), sd_b[k].cpu())):
            return False
    return True


# ----------------------------------------------------------------------------- score
def patches(f, agg):
    if agg > 1:
        f = F.avg_pool2d(f, agg, 1, agg // 2)
    return f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1])  # (B, P, C)


def global_vec(f):
    return torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)  # (B, 2C)


def min_sqdist(q, bank, bank_sq, mem_elems):
    cb = max(1024, int(mem_elems // max(1, q.shape[0])))
    q_sq = (q * q).sum(1, keepdim=True)
    best = torch.full((q.shape[0],), float("inf"), device=q.device)
    for i in range(0, bank.shape[0], cb):
        d = q_sq - 2 * q @ bank[i:i + cb].T + bank_sq[None, i:i + cb]
        best = torch.minimum(best, d.min(1).values)
    return best.clamp_min(0)


def compute_scores(model, layers, train_paths, test_paths, tfm, args, rng):
    """Punteggi per layer: dict layer -> {'patch': (n_test,), 'global': (n_test,)}"""
    dev = args.device
    bank_p = {l: [] for l in layers}
    bank_g = {l: [] for l in layers}
    for x in batches(train_paths, tfm, args.bs, dev):
        fs = extract(model, x, layers)
        for l, f in fs.items():
            if args.mode in ("patch", "both"):
                bank_p[l].append(patches(f, args.agg).reshape(-1, f.shape[1]))
            if args.mode in ("global", "both"):
                bank_g[l].append(global_vec(f))
    out = {l: {} for l in layers}
    banks = {}
    for l in layers:
        if bank_p[l]:
            b = torch.cat(bank_p[l])
            if args.bank_max and b.shape[0] > args.bank_max:
                idx = torch.as_tensor(rng.choice(b.shape[0], args.bank_max, replace=False), device=b.device)
                b = b[idx]
            banks[(l, "patch")] = (b, (b * b).sum(1))
        if bank_g[l]:
            b = torch.cat(bank_g[l])
            banks[(l, "global")] = (b, (b * b).sum(1))
    sc = {(l, m): [] for l in layers for m in ("patch", "global") if (l, m) in banks}
    for x in batches(test_paths, tfm, args.bs, dev):
        fs = extract(model, x, layers)
        for l, f in fs.items():
            if (l, "patch") in banks:
                q = patches(f, args.agg)
                b, bsq = banks[(l, "patch")]
                for i in range(q.shape[0]):
                    sc[(l, "patch")].append(min_sqdist(q[i], b, bsq, args.mem_elems).max().sqrt().item())
            if (l, "global") in banks:
                q = global_vec(f)
                b, bsq = banks[(l, "global")]
                for i in range(q.shape[0]):
                    sc[(l, "global")].append(min_sqdist(q[i:i + 1], b, bsq, args.mem_elems).sqrt().item())
    for (l, m), v in sc.items():
        out[l][m] = np.asarray(v)
    return out


# ----------------------------------------------------------------------------- metriche
def auroc(score, grp, pos, neg, idx=None):
    if idx is not None:
        score, grp = score[idx], grp[idx]
    m = np.isin(grp, [pos] + neg)
    y = (grp[m] == pos).astype(int)
    return roc_auc_score(y, score[m])


def metric_groups(name):
    return {"DvGP": ("defect", ["good", "dust"]),
            "DvP": ("defect", ["dust"]),
            "DvG": ("defect", ["good"])}[name]


def strat_boot(grp, B, rng):
    parts = [np.where(grp == g)[0] for g in ("defect", "good", "dust")]
    for _ in range(B):
        yield np.concatenate([rng.choice(p, len(p), replace=True) for p in parts if len(p)])


# ----------------------------------------------------------------------------- sanity
@torch.no_grad()
def sanity(models, paths, tfm, args):
    """Peso relativo cambiato nel layer3 e coseno tra le feature del layer3 (ImageNet vs Teacher)."""
    base = models["ImageNet"]
    xs = next(batches(paths[:64], tfm, 64, args.device))
    fb = extract(base, xs, [3])[3].flatten(1)
    print("\n[controllo] Teacher caricati correttamente? Confronta con i valori della tua analisi "
          "(variazione conv layer3 ~ 21%, coseno layer3 0,997 / 0,993 / 0,836 per lam 20 / 5 / 0)")
    for tag, m in models.items():
        if tag == "ImageNet":
            continue
        rel = []
        for (kb, vb), (kt, vt) in zip(base.state_dict().items(), m.state_dict().items()):
            if kb.startswith("layer3.") and kb.endswith("weight") and vb.dim() == 4:
                rel.append(((vt - vb).norm() / vb.norm()).item())
        ft = extract(m, xs, [3])[3].flatten(1)
        cos = F.cosine_similarity(fb, ft, dim=1).mean().item()
        print(f"  {tag:>8}: variazione relativa media conv layer3 = {100 * np.mean(rel):5.2f}%   "
              f"coseno medio layer3 = {cos:.4f}")


# ----------------------------------------------------------------------------- tabella LaTeX
def fmt(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def enc_label(tag):
    if tag == "ImageNet_BN":
        return "ImageNet + BN adattata"
    mm = re.fullmatch(r"lam(\d+)", tag)
    return rf"$\lambda_{{\mathrm p}}={mm.group(1)}$" if mm else tag.replace("_", r"\_")


def sgn(x):
    return ("$+$" if x >= 0 else "$-$") + fmt(abs(x))


def write_tex(path, res, deltas, versions, layers, tags, mode):
    rows = []
    for v in versions:
        first = True
        nrow = 0
        block = []
        for l in layers:
            shared = res[(v, mode, l)]["shared"]
            enc_list = ["tutti"] if shared else tags
            for e in enc_list:
                key = "ImageNet" if e == "tutti" else e
                r = res[(v, mode, l)][key]
                cells = [fmt(100 * r["DvGP"][0]), fmt(100 * r["DvP"][0])]
                for m in ("DvGP", "DvP"):
                    d = deltas.get((v, mode, l, key, m))
                    if d is None:
                        cells.append("--")
                    else:
                        dag = r"$^{\dagger}$" if (d[1] > 0 or d[2] < 0) else ""
                        cells.append(f"{sgn(100 * d[0])} [{sgn(100 * d[1])}; {sgn(100 * d[2])}]{dag}")
                lab = "tutti (invariato)" if e == "tutti" else ("ImageNet" if e == "ImageNet" else enc_label(e))
                block.append(f"layer{l} & {lab} & " + " & ".join(cells) + r" \\")
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{v}}}}} \\\\")
        rows.extend(block)
        rows.append(r"\hline")
    tex = (r"""\begin{table}[tb!]
\centering
\caption{Separabilità diretta delle feature (score stile PatchCore, nessuno Student). I-AUROC in percentuale (gli intervalli al 95\% delle AUROC sono nel CSV). DvGP: difettose contro good e polvere. DvP: difettose contro polvere. $\Delta$: differenza dall'ImageNet, con intervallo bootstrap appaiato al 95\% tra parentesi quadre. $\dagger$: intervallo che non contiene zero.}\label{tab:sp-tl-separabilita}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\hline
Layer & Encoder & I-AUROC DvGP & I-AUROC DvP & $\Delta$ DvGP & $\Delta$ DvP \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}}
\end{table}
""")
    Path(path).write_text(tex, encoding="utf8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--version", action="append", help="NOME=CARTELLA (default V1 e V2 del tessuto nero)")
    ap.add_argument("--dust", action="append", required=True, help="[NOME=]file.txt|cartella con la polvere del test")
    ap.add_argument("--teacher", action="append", required=True, help="TAG=checkpoint (es. lam0=...pth)")
    ap.add_argument("--imagenet-weights", default=None, help="state_dict ImageNet locale (default: torchvision IMAGENET1K_V1)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--agg", type=int, default=3, help="finestra della media locale (0 = nessuna)")
    ap.add_argument("--mode", choices=["patch", "global", "both"], default="patch")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bank-max", type=int, default=0, help="tetto alle patch della banca (0 = tutte)")
    ap.add_argument("--mem-elems", type=float, default=2e8, help="elementi massimi per blocco di distanze")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bn-control", action="store_true",
                    help="aggiunge il controllo 'ImageNet_BN': pesi ImageNet con le statistiche della batch normalization "
                         "del layer3 ricalcolate sul train/good della prima versione (nessun apprendimento)")
    ap.add_argument("--out", default="risultati_separabilita")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    versions = parse_kv(args.version) or DEFAULT_VERSIONS
    dust_arg = parse_kv(args.dust, default_key="*")
    tfm = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    # modelli
    if args.imagenet_weights:
        base = make_model(extract_backbone_state(torch_load(args.imagenet_weights)))
    else:
        base = make_model(imagenet=True)
    base = base.to(args.device)
    models = {"ImageNet": base}
    for tag, p in parse_kv(args.teacher).items():
        obj = torch_load(p)
        st = extract_backbone_state(obj)
        if not _has_layers(st):
            print(f"\n{p}: non riconosco i pesi della ResNet-18. Struttura del file:")
            describe(obj)
            sys.exit("Mandami questo elenco: adatto il caricamento.")
        models[tag] = make_model(st, base=base).to(args.device)
    if args.bn_control:
        v0, f0 = next(iter(versions.items()))
        tr0 = list_images(Path(args.data_root) / f0 / "train" / "good")
        ctrl = copy.deepcopy(base)
        for mod in ctrl.layer3.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.reset_running_stats()
                mod.momentum = None  # media cumulativa su tutto il train
                mod.train()
        with torch.no_grad():
            for x in batches(tr0, tfm, args.bs, args.device):
                h = ctrl.maxpool(ctrl.relu(ctrl.bn1(ctrl.conv1(x))))
                ctrl.layer3(ctrl.layer2(ctrl.layer1(h)))
        models["ImageNet_BN"] = ctrl.eval()
        print(f"[controllo] ImageNet_BN: statistiche BN del layer3 ricalcolate su {len(tr0)} immagini di train/good di {v0}")
    tags = list(models)
    sd_base = {k: v for k, v in base.state_dict().items()}
    reuse = {}  # (tag, layer) -> True se identico all'ImageNet fino a quel layer
    for t in tags[1:]:
        sdt = models[t].state_dict()
        for l in args.layers:
            reuse[(t, l)] = same_upto(sdt, sd_base, l)
            print(f"[controllo] {t} layer{l}: " + ("pesi identici all'ImageNet, punteggi riusati" if reuse[(t, l)] else "pesi diversi, calcolo i punteggi"))

    res, deltas, long_rows, delta_rows = {}, {}, [], []
    first_train = None
    for v, folder in versions.items():
        root = Path(args.data_root) / folder
        train = [p for p in list_images(root / "train" / "good")]
        test_good = list_images(root / "test" / "good")
        test_def = [p for p in list_images(root / "test") if p.parent.name != "good"]
        dust_spec = dust_arg.get(v, dust_arg.get("*"))
        if dust_spec is None:
            sys.exit(f"Manca --dust per {v}")
        dust_names = load_dust_names(dust_spec)
        test = test_def + test_good
        grp = np.array(["defect"] * len(test_def) + ["dust" if p.name in dust_names else "good" for p in test_good])
        n_d = int((grp == "dust").sum())
        print(f"\n== {v} ({folder}): train {len(train)}, test difettose {len(test_def)}, "
              f"good {int((grp == 'good').sum())}, polvere {n_d}")
        if n_d != 41:
            print(f"   ATTENZIONE: attese 41 immagini di polvere nel test, trovate {n_d}. Controlla la lista --dust.")
        if n_d == 0:
            sys.exit("Nessuna immagine di polvere riconosciuta: controlla i nomi nella lista --dust.")
        if first_train is None:
            first_train = train
            sanity(models, train, tfm, args)

        # punteggi: ImageNet su tutti i layer, Teacher solo sui layer che cambiano
        scores = {}
        scores["ImageNet"] = compute_scores(base, args.layers, train, test, tfm, args, rng)
        for t in tags[1:]:
            need = [l for l in args.layers if not reuse[(t, l)]]
            got = compute_scores(models[t], need, train, test, tfm, args, rng) if need else {}
            scores[t] = {l: (got[l] if l in got else scores["ImageNet"][l]) for l in args.layers}
        # salvataggio punteggi
        with open(out / f"punteggi_{v}.csv", "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            cols = [(e, l, m) for e in tags for l in args.layers for m in scores[e][l]]
            w.writerow(["immagine", "gruppo"] + [f"{e}_L{l}_{m}" for e, l, m in cols])
            for i, p in enumerate(test):
                w.writerow([str(p), grp[i]] + [f"{scores[e][l][m][i]:.6f}" for e, l, m in cols])

        # metriche e bootstrap appaiato
        boots = list(strat_boot(grp, args.boot, rng))
        for mode in (["patch", "global"] if args.mode == "both" else [args.mode]):
            for l in args.layers:
                shared = all(reuse[(t, l)] for t in tags[1:])
                res[(v, mode, l)] = {"shared": shared}
                for e in tags:
                    s = scores[e][l][mode]
                    r = {}
                    for mname, _ in METRICS:
                        pos, neg = metric_groups(mname)
                        a = auroc(s, grp, pos, neg)
                        bs = np.array([auroc(s, grp, pos, neg, idx) for idx in boots])
                        r[mname] = (a, *np.percentile(bs, [2.5, 97.5]))
                        long_rows.append([v, mode, l, e, mname, a, r[mname][1], r[mname][2]])
                    res[(v, mode, l)][e] = r
                if shared:
                    continue
                for t in tags[1:]:
                    if reuse[(t, l)]:
                        continue
                    for mname, _ in METRICS:
                        pos, neg = metric_groups(mname)
                        a_t = auroc(scores[t][l][mode], grp, pos, neg)
                        a_i = auroc(scores["ImageNet"][l][mode], grp, pos, neg)
                        db = np.array([auroc(scores[t][l][mode], grp, pos, neg, idx) -
                                       auroc(scores["ImageNet"][l][mode], grp, pos, neg, idx) for idx in boots])
                        lo, hi = np.percentile(db, [2.5, 97.5])
                        deltas[(v, mode, l, t, mname)] = (a_t - a_i, lo, hi)
                        delta_rows.append([v, mode, l, t, mname, a_t - a_i, lo, hi, int(lo > 0)])

    with open(out / "separabilita.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["versione", "modo", "layer", "encoder", "metrica", "auroc", "ci_lo", "ci_hi"])
        w.writerows(long_rows)
    with open(out / "delta_vs_imagenet.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["versione", "modo", "layer", "teacher", "metrica", "delta", "ci_lo", "ci_hi", "estremo_inferiore_positivo"])
        w.writerows(delta_rows)
    for mode in (["patch", "global"] if args.mode == "both" else [args.mode]):
        write_tex(out / f"tabella_separabilita_{mode}.tex", res, deltas, list(versions), args.layers, tags, mode)

    # riepilogo e verdetto
    print("\n================ RIEPILOGO ================")
    for mode in (["patch", "global"] if args.mode == "both" else [args.mode]):
        print(f"\n-- modo: {mode}")
        for v in versions:
            print(f" {v}  I-AUROC ImageNet per layer (DvGP / DvP):")
            for l in args.layers:
                r = res[(v, mode, l)]["ImageNet"]
                print(f"   layer{l}: DvGP {100 * r['DvGP'][0]:5.1f} [{100 * r['DvGP'][1]:5.1f}; {100 * r['DvGP'][2]:5.1f}]   "
                      f"DvP {100 * r['DvP'][0]:5.1f} [{100 * r['DvP'][1]:5.1f}; {100 * r['DvP'][2]:5.1f}]")
            for t in tags[1:]:
                key_a = (v, mode, 3, t, "DvGP")
                key_b = (v, mode, 3, t, "DvP")
                if key_a not in deltas:
                    print(f"   {t:>8} layer3: pesi identici all'ImageNet, nessun confronto da fare "
                          f"(controlla che il checkpoint sia quello giusto)")
                    continue
                da, db_ = deltas[key_a], deltas[key_b]
                ok = da[1] > 0 and db_[1] > 0
                print(f"   {t:>8} layer3: Δ DvGP {100 * da[0]:+5.2f} [{100 * da[1]:+5.2f}; {100 * da[2]:+5.2f}]   "
                      f"Δ DvP {100 * db_[0]:+5.2f} [{100 * db_[1]:+5.2f}; {100 * db_[2]:+5.2f}]   -> {'controllo BN' if t == 'ImageNet_BN' else 'Teacher utile'}: {'SI' if ok else 'NO'}")
    print(f"\nFile in {out}/: separabilita.csv, delta_vs_imagenet.csv, punteggi_<versione>.csv, tabella_separabilita_<modo>.tex")


if __name__ == "__main__":
    main()