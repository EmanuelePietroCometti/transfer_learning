import torch 
from torchvision import models
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
import os
from sklearn.metrics import roc_auc_score

def train(ckpt_path, train_dataset, val_dataset, writer, batch_size=16, device="cuda", lr=1e-4, epochs=15, seed=42, num_workers=4, print_report=5):
    torch.manual_seed(seed)

    # Instanziamento del dataloader con batch size e mescolamento nel caso del train
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # Caricamento backbone
    model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2) # Transfer learning per adattare il modello al numero di classi della nostra task
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr)

    best_auroc = -1.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss/len(train_dataloader)
        print(f"Epoch: {epoch} - Loss: {avg_loss}")


        if epoch % print_report == 0:
            val_preds, val_labels, val_y_probs = [],[],[]

            model.eval()
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    logits = model(images)
                    _, pred = torch.max(logits, 1)

                    # Prendo le probabilità di predizione per la classe positiva
                    probs = torch.softmax(logits, dim=1)[:, 1]

                    # Estendo la lista con i risultati del batch
                    val_preds.extend(pred.tolist())
                    val_labels.extend(labels.tolist())
                    val_y_probs.extend(probs.tolist())


            # Calcolo delle metriche di classificazione
            report_dict = classification_report(val_labels, val_preds, output_dict=True)

            # Calcolo dell'AUROC usando le labels e le probabilità
            auroc = roc_auc_score(val_labels, val_y_probs)

            acc = report_dict["accuracy"]
            prec = report_dict["weighted avg"]["precision"]
            rec = report_dict["weighted avg"]["recall"]
            f1 = report_dict["weighted avg"]["f1-score"]

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

            if auroc > best_auroc:
                best_auroc = auroc
                # Salvataggio del backbone senza fc e head 
                sd = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}

                os.makedirs(ckpt_path, exist_ok=True)
                file_path = os.path.join(ckpt_path, "best_encoder.pth")

                # Salvataggio del modello
                torch.save({"encoder": sd}, file_path)
                print(f"  -> Saved {file_path} (AUROC {auroc:.4f})")

    writer.close()
    return best_auroc, model

