import torch 
from torchvision import models
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from sklearn.metrics import classification_report
import os
from sklearn.metrics import roc_auc_score


def set_bn_eval(model):
    # running stats delle BN congelate; gamma/beta restano addestrabili
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def set_frozen_eval(model, frozen):
    # i moduli congelati restano in eval(): requires_grad=False non ferma l'aggiornamento delle running stats delle BN
    for name in frozen:
        getattr(model, name).eval()

def layer3_features(m, x):
    # stessa forward di torchvision ResNet fino al layer3 (le feature che SK-RD4AD usa insieme a layer1-2)
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    return m.layer3(m.layer2(m.layer1(x)))

def build_param_groups(model, lr, backbone_lr_mult, weight_decay, l2sp):
    # testa con lr pieno, backbone con lr * backbone_lr_mult, niente decay su BN e bias
    head, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head.append(p)
        elif p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    backbone_lr = lr * backbone_lr_mult
    return [
        {"params": head, "lr": lr, "weight_decay": weight_decay},
        {"params": decay, "lr": backbone_lr, "weight_decay": 0.0 if l2sp > 0 else weight_decay},
        {"params": no_decay, "lr": backbone_lr, "weight_decay": 0.0},
    ]

def train(ckpt_path, train_dataset, val_dataset, writer, batch_size=16, device="cuda", lr=1e-4, epochs=15, seed=42, num_workers=4, print_report=5, freeze_bn=False, backbone_lr_mult=1.0, weight_decay=0.01, l2sp=0.0, freeze_layers=(), head_on_layer3=False, lam_preserve=0.0):
    torch.manual_seed(seed)

    # Instanziamento del dataloader con batch size e mescolamento nel caso del train
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Caricamento backbone
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    if head_on_layer3:
        # il layer4 non viene usato da SK-RD4AD: lo tolgo e metto la testa (max-pool globale + linear) sul layer3
        model.layer4 = nn.Identity()
        model.avgpool = nn.AdaptiveMaxPool2d(1)
        model.fc = nn.Linear(256, 2)
    else:
        model.fc = nn.Linear(model.fc.in_features, 2) # Transfer learning per adattare il modello al numero di classi della nostra task
    model = model.to(device)

    # Congelamento: pesi fissi (le BN dei moduli congelati vengono messe in eval nel loop)
    for name, p in model.named_parameters():
        if name.split(".")[0] in freeze_layers:
            p.requires_grad = False

    # Output del layer3 catturato con un hook (serve alla loss di preservazione)
    feats = {}
    model.layer3.register_forward_hook(lambda m, i, o: feats.__setitem__("l3", o))

    # Riferimento ImageNet per la loss di preservazione (solo forward, nessun gradiente)
    ref = None
    if lam_preserve > 0:
        ref = models.resnet18(weights = models.ResNet18_Weights.DEFAULT).to(device).eval()
        for p in ref.parameters():
            p.requires_grad = False

    os.makedirs(ckpt_path, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(build_param_groups(model, lr, backbone_lr_mult, weight_decay, l2sp))

    # Pesi ImageNet di riferimento per L2-SP (solo i pesi conv della backbone)
    anchors = [(p, p.detach().clone()) for n, p in model.named_parameters()
               if l2sp > 0 and not n.startswith("fc.") and p.ndim > 1]


    best_score = (-1.0, float("-inf"))
    best_auroc = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()

        if freeze_bn:
            set_bn_eval(model)
        set_frozen_eval(model, freeze_layers)

        running_loss = 0.0
        running_pres = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            if ref is not None:
                with torch.no_grad():
                    f0 = layer3_features(ref, images)
                pres = (1 - F.cosine_similarity(feats["l3"], f0, dim=1)).mean()
                loss = loss + lam_preserve * pres
                running_pres += pres.item()

            loss.backward()
            optimizer.step()

             # richiamo verso i pesi ImageNet
            if anchors:
                with torch.no_grad():
                    lr_backbone = optimizer.param_groups[1]["lr"]
                    for p, p0 in anchors:
                        p.sub_(lr_backbone * l2sp * (p - p0))

            running_loss += loss.item()

        avg_loss = running_loss/len(train_dataloader)
        writer.add_scalar("Avg loss", avg_loss, epoch)
        msg = f"Epoch: {epoch} - Loss: {avg_loss:.4f}"
        if ref is not None:
            avg_pres = running_pres/len(train_dataloader)
            writer.add_scalar("Preservation (1-cos)", avg_pres, epoch)
            msg += f" - 1-cos layer3 vs ImageNet: {avg_pres:.4f}"
        print(msg)

        # Checkpoint di ogni epoca: servono per la curva epoche -> AUROC di SK-RD4AD
        sd_ep = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("fc.")}
        torch.save({"encoder": sd_ep, "epoch": epoch}, os.path.join(ckpt_path, f"encoder_ep{epoch + 1:02d}.pth"))

        if (epoch+1) % print_report == 0 or epoch == epochs-1:
            val_preds, val_labels, val_y_probs = [],[],[]
            val_loss = 0.0

            model.eval()
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    logits = model(images)
                    val_loss += loss_fn(logits, labels).item() * labels.size(0)
                    _, pred = torch.max(logits, 1)

                    # Prendo le probabilità di predizione per la classe positiva
                    probs = torch.softmax(logits, dim=1)[:, 1]

                    # Estendo la lista con i risultati del batch
                    val_preds.extend(pred.tolist())
                    val_labels.extend(labels.tolist())
                    val_y_probs.extend(probs.tolist())

            val_loss /= len(val_labels)
            writer.add_scalar("Val/Loss", val_loss, epoch)

            # Calcolo delle metriche di classificazione
            report_dict = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)

            # Calcolo dell'AUROC usando le labels e le probabilità
            auroc = roc_auc_score(val_labels, val_y_probs)

            acc = report_dict["accuracy"]
            prec = report_dict["macro avg"]["precision"]
            rec = report_dict["macro avg"]["recall"]
            f1 = report_dict["macro avg"]["f1-score"]

            # Stampo le metriche di predizione su TensorBoard
            writer.add_scalar("Val/Accuracy", acc, epoch)
            writer.add_scalar("Val/Precision", prec, epoch)
            writer.add_scalar("Val/Recall", rec, epoch)
            writer.add_scalar("Val/F1-score", f1, epoch)
            writer.add_scalar("Val/AUROC", auroc, epoch)

            print(
                f"[Epoch {epoch}] Validation Report:\n"
                f" - Accuracy:  {acc:.2%}\n"
                f" - Precision: {prec:.2%}\n"
                f" - Recall:    {rec:.2%}\n"
                f" - F1-Score:  {f1:.2%}\n"
                f" - AUROC:     {auroc:.4f}"
            )

            score = (auroc, -val_loss)
            if score > best_score:
                best_score = score
                best_auroc = auroc
                # Salvataggio del backbone senza fc e head 
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                sd = {k: v for k, v in best_state.items() if not k.startswith("fc.")}
                fc = {k: v for k, v in best_state.items() if k.startswith("fc.")}

                os.makedirs(ckpt_path, exist_ok=True)
                file_path = os.path.join(ckpt_path, "best_encoder.pth")

                # Salvataggio del modello
                torch.save({"encoder": sd, "fc": fc, "epoch": epoch, "val_auroc": auroc, "val_loss": val_loss}, file_path)
                print(f"  -> Saved {file_path} (AUROC {auroc:.4f})")

    writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auroc, model

