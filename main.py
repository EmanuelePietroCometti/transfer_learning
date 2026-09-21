from torchvision import datasets, transforms
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from .src.train import train

def main():
    parser = argparse.ArgumentParser(
            prog="Transfer learning per dataset tessuti",
            description="Argomenti di configurazione dell'addestramento"
        )
    # Dati di input
    parser.add_argument("--ckpt_path", type=str, default="checkpoint", description="Path alla cartella in cui salvare il checkpoint del modello")
    parser.add_argument("--dataset_path", type=str, default="dataset", description="Path alla cartella usata per l'addestramento")
    parser.add_argument("--output_dir", type=str, default="dataset", description="Path alla cartella in cui salvare i risultati")

    # Parametri di configurazione dell'addestramento
    parser.add_argument("--lr", type=float, default=1e-4, description="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, description="Numero di epoche per cui eseguire l'addestramento")
    parser.add_argument("--batch_size", type=int, default=16, default="Dimensione del batch size usato durante l'addestramento")
    parser.add_argument("--device", type=str, default="cuda", description="Device da usare per l'accelerazione dell'addestramento")
    parser.add_argument("--train_rate", type=float, default=0.7, description="Percentuale di immagini da usare per lo split di train")
    parser.add_argument("--val_rate", type=float, default=0.2, description="Percentuali di immagini da usare per lo split di validation")
    parser.add_argument("--test_rate", type=float, default=0.1, description="Percentuali di immagini da usare per lo split di test")

    # Seed per la riproducibilità degli esperimenti
    parser.add_argument("--seed", type=int, default=42, description="Random seed per riproducibilità")

    parser.add_argument("--num_workers", type=int, default=4, description="Numero di workers per il dataloader")

    args = parser.parse_args()

    writer = SummaryWriter()

    # Trasformazioni da applicare a tutte le immagini del dataset
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Caricamento del dataset con divisione in cartelle
    dataset = datasets.ImageFolder(root=args.dataset_path, transform=transform)

    # Split del dataset
    train_size = len(dataset) * args.train_rate
    val_size = len(dataset) * args.val_rate
    test_size = len(dataset) * args.test_rate
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], seed=args.seed)

    _, model = train(ckpt_path=args.ckpt_path, train_dataset=train_dataset, val_dataset=val_dataset, writer=writer, batch_size=args.batch_size, device=args.device, lr=args.lr, epochs=args.epochs, seed=args.seed, num_workers=args.num_workers)

    eval(test_dataset, output_dir=args.output_dir, model=model, batch_size=args.batch_size, num_workers=args.num_workers, device=args.device)



if __name__ == "__main__":
    main()