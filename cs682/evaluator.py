import torch

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
    print("\n")
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
