import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os

from dataset import split_task, get_dataset_new_task, get_loader
from classifier.AlexNet.AlexNet import get_new_task_classifier


def task_classifier_train(
    model, dataloader, argsimizer, epochs = 200, device="cpu", patience=5
):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    progress_bar = tqdm(range(epochs), desc="Training Progress", leave=True)
    best_accuracy = 0.0
    no_improvement_epochs = 0

    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            argsimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            argsimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        # Update progress bar
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.4f}", accuracy=f"{accuracy:.2f}%")

        # Early stopping check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_epochs = 0  # Reset counter
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"\nEarly stopping triggered. Best accuracy: {best_accuracy:.2f}%")
            break

    print("Training completed.")
    return model

# def eval_classifier():