import torch
import torch.nn as nn
from tqdm import tqdm
import os
from collections import defaultdict

from copy import deepcopy

from dataset import get_dataset_new_task, get_loader
from visualize import plot_task_vs_accuracy


def task_classifier_train(
    model:nn.Module, dataloader, optimzier, 
    manager, val_dataset_name, val_class_indicies, val_epoch_step=10, 
    epochs = 100, device="cuda"
):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_eval_acc = 0
    best_model = deepcopy(model)

    progress_bar = tqdm(range(epochs), desc="Training Progress", leave=True)

    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, _, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimzier.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimzier.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        if epoch % val_epoch_step == 0 or epoch == epochs-1:
            eval_acc, _ = _eval_classifier(
                model=model, 
                dataset_name=val_dataset_name, 
                class_indicies=val_class_indicies, 
                saver=manager, device=device)
            
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_model = deepcopy(model)

        # Update progress bar
        progress_bar.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.4f}", accuracy=f"{accuracy:.2f}%", last_eval_acc=f"{eval_acc:.2f}%")

    model.load_state_dict(best_model.state_dict())

    print("Classifier Training completed.")
    return model

def eval_classifier(model, dataset_name, class_idx_lst, saver, device = 'cpu'):
    accs = []
    for task, class_indicies in enumerate(class_idx_lst):
        acc, _ = _eval_classifier(model, dataset_name, class_indicies, saver, device)
        accs.append(acc)
        print(f'task {task} accuracy {acc}')

    # Save accs to a text file
    accs_txt_path = os.path.join(saver.get_results_path(), 'task_accuracies.txt')
    with open(accs_txt_path, 'w') as f:
        for task, acc in enumerate(accs):
            f.write(f"Task {task}: {acc}\n")
    print(f"Accuracy values saved to {accs_txt_path}")

    save_path = os.path.join(saver.get_results_path(), 'plot_task_vs_accuracy.png')
    plot_task_vs_accuracy(range(len(class_idx_lst)), accs, save_path)

def _eval_classifier(model, dataset_name, class_indicies, saver, device):
    _, test_dataset = get_dataset_new_task(dataset_name = dataset_name, class_indicies = class_indicies)
    loader = get_loader([test_dataset], batch_size=16, curr_classes=None, saver=saver, test=True)

    model.eval()

    total_correct = 0
    total_samples = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, _, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
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