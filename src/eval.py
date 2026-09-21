from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch
import os

def eval(test_dataset, output_dir, model, batch_size, num_workers, device="cuda"):
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_preds, val_labels, val_y_probs = [],[],[]
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in test_dataloader:
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

            for i in range(labels):
                true_label = labels[i].item()
                pred_label = pred[i].item()
                img_tensor = images[i].cpu()

                # Determino la cartella di appartenenza basandomi sulla confusion matrix
                if true_label == 1 and pred_label == 1:
                    category = "TP"
                elif true_label == 0 and pred_label == 0:
                    category = "TN"
                elif true_label == 0 and pred_label == 1:
                    category = "FP"
                elif true_label == 1 and pred_label == 0:
                    category = "FN"

                filename = f"img_{batch_idx * batch_size + i}.png"
                save_path = os.path.join(output_dir, category, filename)

                save_image(img_tensor, save_path)

    # Calcolo delle metriche di classificazione
    report_dict = classification_report(val_labels, val_preds, output_dict=True)

    # Calcolo dell'AUROC usando le labels e le probabilità
    auroc = roc_auc_score(val_labels, val_y_probs)

    # Calcolo della matrice di confusione
    cm = confusion_matrix()

    acc = report_dict["accuracy"]
    prec = report_dict["precision"]
    rec = report_dict["recall"]
    f1 = report_dict["f1-score"]

    print(
        f"Test Report:\n"
        f" - Accuracy:  {acc:.2%}\n"
        f" - Precision: {prec:.2%}\n"
        f" - Recall:    {rec:.2%}\n"
        f" - F1-Score:  {f1:.2%}\n"
        f" - AUROC:     {auroc:.4f}"
    )

    plt.figure(figsize=(6,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("Matrice di Confusione")
    plt.ylabel("Valori Reali")
    plt.xlabel("Valori Predetti")
    plt.show()

    

