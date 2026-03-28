import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import json

from models.teacher import BERTTeacher
from data.loader import IMDBDataset, YelpDataset, AmazonDataset

from argparse import ArgumentParser

class TeacherTrainConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"TeacherTrainConfig({self.__dict__})"

def check_accuracy(model: BERTTeacher, loader: DataLoader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted = torch.argmax(out["logits"], dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

def train(config: TeacherTrainConfig, train_loader: DataLoader, validation_loader: DataLoader):
    print(config)

    criterion = config.criterion
    optimizer = config.optimizer
    model = config.model

    epochs = config.epochs
    log_every_k = config.log_every_k
    accuracy_every_k = config.accuracy_every_k

    device = config.device

    model.to(device)

    losses = []
    train_accuracies = []
    validation_accuracies = []

    step = 0
    for epoch in range(epochs):
        model.train()

        for i, batch in enumerate(train_loader):
            step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(out["logits"], targets)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if step % log_every_k == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {step}, Batch {i}/{len(train_loader)}, Loss: {losses[-1]:.4f}")

            if step % accuracy_every_k == 0:
                train_acc = check_accuracy(model, train_loader, device)
                val_acc = check_accuracy(model, validation_loader, device)
                train_accuracies.append(train_acc)
                validation_accuracies.append(val_acc)
                print(f"  Train Acc: {train_acc:.4f}  |  Val Acc: {val_acc:.4f}")
                model.train()

    return model, losses, train_accuracies, validation_accuracies

def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate_fn

if __name__ == "__main__":
    arg = ArgumentParser()
    arg.add_argument("--dataset", choices=["imdb", "yelp", "amazon"], default="imdb")
    arg.add_argument("--epochs", type=int, default=3)
    arg.add_argument("--learning_rate", type=float, default=2e-5)
    arg.add_argument("--validation_pct", type=float, default=0.2)
    arg.add_argument("--batch_size", type=int, default=32)
    arg.add_argument("--log_every_k", type=int, default=50)
    arg.add_argument("--accuracy_every_k", type=int, default=200)
    arg.add_argument("--max_length", type=int, default=128)
    arg.add_argument("--dropout", type=float, default=0.1)
    arg.add_argument("--model_save_path", type=str, default="teacher.pt")
    arg.add_argument("--train_save_path", type=str, default="teacher.json")
    arg.add_argument("--mapped_layer_indices", type=int, nargs="+", default=[4, 8])
    args = arg.parse_args()

    print(args)

    num_classes = 5
    if args.dataset == "imdb":
        dataset = IMDBDataset()
        num_classes = 2
    elif args.dataset == "yelp":
        dataset = YelpDataset()
    elif args.dataset == "amazon":
        dataset = AmazonDataset()
    else:
        raise ValueError()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = BERTTeacher.from_pretrained(
        num_classes=num_classes,
        mapped_layer_indices=args.mapped_layer_indices,
        dropout=args.dropout,
    )

    val_size = int(len(dataset) * args.validation_pct)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    train_config = TeacherTrainConfig(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        epochs=args.epochs,
        log_every_k=args.log_every_k,
        accuracy_every_k=args.accuracy_every_k,
        device=device,
    )

    model, losses, train_accs, val_accs = train(train_config, train_loader, val_loader)

    with open(args.train_save_path, "w") as f:
        obj = {
            "losses": losses,
            "train_accs": train_accs,
            "validation_accs": val_accs
        }

        json.dump(obj, f, indent=2)

    torch.save(model.state_dict(), args.model_save_path)
    print(f"\nModel saved to {args.save_path}")
