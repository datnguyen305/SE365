import torch

class Evaluator:
    def __init__(self, model, loss_fn, device="cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def evaluate(self, dataloader):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # for binary F1 (class = 1 là positive)
        tp = fp = fn = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["image"].to(self.device)
                target = batch["label"].to(self.device)

                logits = self.model(x)          # [B, C]
                loss = self.loss_fn(logits, target)

                batch_size = x.size(0)
                total_loss += loss.item() * batch_size

                preds = torch.argmax(logits, dim=1)

                total_correct += (preds == target).sum().item()
                total_samples += batch_size

                # ===== Binary F1 =====
                tp += ((preds == 1) & (target == 1)).sum().item()
                fp += ((preds == 1) & (target == 0)).sum().item()
                fn += ((preds == 0) & (target == 1)).sum().item()

                # 🔥 Lưu để vẽ confusion matrix
                all_preds.append(preds.cpu())
                all_targets.append(target.cpu())

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "y_pred": torch.cat(all_preds).numpy(),
            "y_true": torch.cat(all_targets).numpy(),
        }
