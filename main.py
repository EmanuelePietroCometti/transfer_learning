from torchvision import datasets, transforms
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from src.train import train
from src.eval import evaluate
import os
from src.config import IMG_MEAN, IMG_STD
CLASS_TO_IDX = {"good": 0, "defect": 1}


class GoodDefectFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        found = sorted(e.name for e in os.scandir(directory) if e.is_dir())
        if set(found) != set(CLASS_TO_IDX):
            raise FileNotFoundError(
                f"In {directory} servono esattamente le cartelle {sorted(CLASS_TO_IDX)}, trovate: {found}"
            )
        return list(CLASS_TO_IDX), dict(CLASS_TO_IDX)

def set_deterministic(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(
            prog="Transfer learning per dataset tessuti",
            description="Argomenti di configurazione dell'addestramento"
        )
    # Dati di input
    parser.add_argument("--ckpt_path", type=str, default="checkpoint", help="Path alla cartella in cui salvare il checkpoint del modello")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path alla cartella usata per l'addestramento")
    parser.add_argument("--output_path", type=str, default="output", help="Path alla cartella in cui salvare i risultati")
    parser.add_argument("--project_name", type=str, default="project_name", help="Nome del progetto")


    # Parametri di configurazione dell'addestramento
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Numero di epoche per cui eseguire l'addestramento")
    parser.add_argument("--batch_size", type=int, default=16, help="Dimensione del batch size usato durante l'addestramento")
    parser.add_argument("--device", type=str, default="cuda", help="Device da usare per l'accelerazione dell'addestramento")
    parser.add_argument("--train_rate", type=float, default=0.7, help="Percentuale di immagini da usare per lo split di train")
    parser.add_argument("--val_rate", type=float, default=0.2, help="Percentuali di immagini da usare per lo split di validation")
    parser.add_argument("--test_rate", type=float, default=0.1, help="Percentuali di immagini da usare per lo split di test")
    parser.add_argument("--print_report", type=int, default=5, help="Intervallo ogni quale vengono stampate le metriche di valutazione")
    parser.add_argument("--freeze_bn", type=int, default=0, help="1 = running stats delle BN congelate")
    parser.add_argument("--backbone_lr_mult", type=float, default=1.0, help="LR backbone = lr * mult")
    parser.add_argument("--l2sp", type=float, default=0.0, help="Richiamo verso i pesi ImageNet (0 = disattivo)")
    parser.add_argument("--freeze_layers", type=str, nargs="*", default=[], help="Moduli da congelare (pesi + BN in eval), es.: conv1 bn1 layer1 layer2")
    parser.add_argument("--head_on_layer3", type=int, default=0, help="1 = rimuove il layer4 e mette la testa (max-pool + linear) sul layer3")
    parser.add_argument("--lam_preserve", type=float, default=0.0, help="Peso della loss di preservazione delle feature del layer3 rispetto a ImageNet (0 = disattiva)")

    # Seed per la riproducibilità degli esperimenti
    parser.add_argument("--seed", type=int, default=42, help="Random seed per riproducibilità")
    parser.add_argument("--split_seed", type=int, default=0, help="Seed dello split train/val/test: tienilo uguale in tutti i run")

    parser.add_argument("--num_workers", type=int, default=4, help="Numero di workers per il dataloader")

    args = parser.parse_args()

    # Imposto seed per determinismo
    set_deterministic(args.seed)

    run_output_path = os.path.join(args.output_path, args.project_name, f"seed_{args.seed}")
    run_ckpt_path = os.path.join(run_output_path, args.ckpt_path)

    # Assicuro che le directory esistano
    os.makedirs(args.dataset_path, exist_ok=True)
    os.makedirs(run_output_path, exist_ok=True)
    os.makedirs(run_ckpt_path, exist_ok=True)

    # Instanzio il writer di TensorBoards
    writer = SummaryWriter()

    # Trasformazioni da applicare a tutte le immagini del dataset
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        # transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])

    # Caricamento del dataset con divisione in cartelle
    dataset = GoodDefectFolder(root=args.dataset_path, transform=transform)
    print("class_to_idx:", dataset.class_to_idx)

    # Split del dataset
    total_size = len(dataset)
    train_size = int(total_size * args.train_rate)
    val_size = int(total_size* args.val_rate)
    test_size = total_size - train_size - val_size
    split_generator = torch.Generator().manual_seed(args.split_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=split_generator)

    _, model = train(ckpt_path=run_ckpt_path, train_dataset=train_dataset, val_dataset=val_dataset, writer=writer, batch_size=args.batch_size, device=args.device, lr=args.lr, epochs=args.epochs, seed=args.seed, num_workers=args.num_workers, print_report=args.print_report, freeze_bn=bool(args.freeze_bn), backbone_lr_mult=args.backbone_lr_mult, l2sp=args.l2sp, freeze_layers=args.freeze_layers, head_on_layer3=bool(args.head_on_layer3), lam_preserve=args.lam_preserve)

    evaluate(test_dataset, output_dir=run_output_path, model=model, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device)



if __name__ == "__main__":
    main()