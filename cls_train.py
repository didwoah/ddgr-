import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
from collections import defaultdict

from dataset import split_task, get_dataset_new_task, get_loader
from visualize import plot_task_vs_accuracy


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

def eval_classifier(model, dataset_name, prev_task_class_lst, map_path, saver):
    now_task = len(prev_task_class_lst) - 1
    accs = []
    for task, class_lst in enumerate(prev_task_class_lst):
        acc, _ = _eval_classifier(dataset_name, class_lst, model, map_path)
        accs.append(acc)
        print(f'task {now_task} -> task {task} accuracy {acc}')

    # 김민석의 구현

    save_path = saver.get_plot_path()

    plot_task_vs_accuracy(range(len(prev_task_class_lst)), accs, save_path)

def _eval_classifier(model, dataset_name, class_lst, map_path):
    _, test_dataset = get_dataset_new_task(dataset_name, class_lst)
    loader = get_loader(test_dataset, batch_size=1, map_path=map_path)

    model.eval()

    total_correct = 0
    total_samples = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Update total metrics
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # Update class-wise metrics
            for target, pred in zip(targets, predicted):
                class_total[target.item()] += 1
                if target == pred:
                    class_correct[target.item()] += 1

    # Calculate total accuracy
    total_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

    # Calculate class-wise accuracy
    class_accuracy = {
        class_label: 100.0 * class_correct[class_label] / class_total[class_label]
        if class_total[class_label] > 0 else 0.0
        for class_label in class_total
    }

    return total_accuracy, class_accuracy