import torch
from torch.utils.data import DataLoader

from models.teacher import BERTTeacher
from data.loader import IMDBDataset, YelpDataset, AmazonDataset
from teacher_trainer import make_collate_fn

from argparse import ArgumentParser


def evaluate(model, loader, num_classes, device):
    print("Starting evaluation.")
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted = torch.argmax(out["logits"], dim=1)

            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())

            if i % 50 == 0:
                print(f"{i} / {len(loader)} Complete")

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    total = len(targets)
    correct = (preds == targets).sum().item()
    accuracy = correct / total
    classification_error = 1.0 - accuracy

    print(f"\nOverall Accuracy:           {accuracy:.4f}")
    print(f"Overall Classification Error: {classification_error:.4f}")
    print()

    print(f"{'Class':<10} {'Precision':>12} {'Recall':>12}")
    print("-" * 36)
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"{c:<10} {precision:>12.4f} {recall:>12.4f}")

    return accuracy, classification_error


if __name__ == "__main__":
    arg = ArgumentParser()
    arg.add_argument("--dataset", choices=["imdb", "yelp", "amazon"], default="imdb")
    arg.add_argument("--split", choices=["train", "test"], default="test")
    arg.add_argument("--model_type", choices=["teacher", "student"], default="teacher")
    arg.add_argument("--checkpoint", type=str, required=True)
    arg.add_argument("--batch_size", type=int, default=32)
    arg.add_argument("--max_length", type=int, default=128)
    arg.add_argument("--dropout", type=float, default=0.1)
    # teacher-specific
    arg.add_argument("--mapped_layer_indices", type=int, nargs="+", default=[4, 8])
    # student-specific
    arg.add_argument("--block_size", type=str, default="2_2_2")
    args = arg.parse_args()

    print(args)

    num_classes = 5
    if args.dataset == "imdb":
        dataset = IMDBDataset(split=args.split)
        num_classes = 2
    elif args.dataset == "yelp":
        dataset = YelpDataset(split=args.split)
    elif args.dataset == "amazon":
        dataset = AmazonDataset(split=args.split)
    else:
        raise ValueError()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "teacher":
        model, tokenizer = BERTTeacher.from_pretrained(
            num_classes=num_classes,
            mapped_layer_indices=args.mapped_layer_indices,
            dropout=args.dropout,
        )

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    evaluate(model, loader, num_classes, device)
