from dataloader import CatDogDataset
from evaluate import Evaluator
from models import NeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import csv
from tqdm import tqdm
import numpy as np
from visualize import plot_confusion_matrix, plot_learning_curves

def train(config):
    ckpt_dir = os.path.dirname(config.train.ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    train_dataset = CatDogDataset(config.dataset, splits="train")
    dev_dataset = CatDogDataset(config.dataset, splits="dev")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.train.batch_size,
        shuffle=False
    )

    model = NeuralNetwork(config.model).to(config.device)
    loss_fn = nn.CrossEntropyLoss()

    if config.train.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=float(config.train.weight_decay)
        )
    
    if config.train.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.train.learning_rate,
            momentum=0.9,
            weight_decay=float(config.train.weight_decay)
        )

    evaluator = Evaluator(model, loss_fn, device=config.device)

    best_dev_loss = float("inf")
    result_per_epoch = []
    patience = config.train.patience   # ví dụ: 3
    counter = 0

    for epoch in range(config.train.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{config.train.epochs}]",
            leave=False
        )

        for batch in pbar:
            x = batch["image"].to(config.device)
            target = batch["label"].to(config.device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size

            # update progress bar
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader.dataset)
        dev_metrics = evaluator.evaluate(dev_loader)

        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = dev_metrics["loss"]
            counter = 0
            torch.save(model.state_dict(), config.train.ckpt_path)
            print(f" -> Best model saved at epoch {epoch+1}")
        else:
            counter += 1

        print(f"Epoch {epoch+1}/{config.train.epochs}, Dev Accuracy: {dev_metrics['accuracy']:.4f}, Dev F1: {dev_metrics['f1']:.4f}")
        
        result_per_epoch.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"]
        })

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        

    csv_path = f"{ckpt_dir}/training_log_{config.train.type}.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "dev_loss", "dev_accuracy"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_per_epoch[-1])

def test(config):
    ckpt_dir = os.path.dirname(config.train.ckpt_path)
    final_results = []
    test_dataset = CatDogDataset(config.dataset, splits="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False
    )

    model = NeuralNetwork(config.model).to(config.device)
    model.load_state_dict(torch.load(config.train.ckpt_path))

    loss_fn = nn.CrossEntropyLoss()
    evaluator = Evaluator(model, loss_fn, device=config.device)

    test_metrics = evaluator.evaluate(
        tqdm(test_loader, desc="Testing")
    )

    y_pred = test_metrics["y_pred"]
    y_true = test_metrics["y_true"]

    np.save(f"{ckpt_dir}/y_pred_{config.train.type}.npy", y_pred)
    np.save(f"{ckpt_dir}/y_true_{config.train.type}.npy", y_true)
    print(f" -> Saved predictions and targets to {ckpt_dir}")

    print(
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Acc: {test_metrics['accuracy']:.4f} | "
        f"Test F1: {test_metrics['f1']:.4f}"
    )
    final_results.append({
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"]
    })
    # Vẽ và lưu confusion matrix
    labels = test_dataset.classes  # ['Cat', 'Dog']
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels,
        save_path=f"{ckpt_dir}/confusion_matrix_{config.train.type}.png",
        title="Confusion Matrix"
    )
    # Vẽ và lưu learning curves
    plot_learning_curves(
        csv_path=f"{ckpt_dir}/training_log_{config.train.type}.csv",
        save_path=f"{ckpt_dir}/learning_curves_{config.train.type}.png"
    )

    print("Final Test Results:")
    for key, value in final_results[-1].items():
        print(f" - {key}: {value:.4f}")

    csv_path = f"{ckpt_dir}/final_result_{config.train.type}.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["accuracy", "precision", "recall", "test_f1"]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(final_results[-1])