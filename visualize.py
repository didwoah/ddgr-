import matplotlib.pyplot as plt

def plot_task_vs_accuracy(task_numbers, accuracies, output_path="task_vs_accuracy.png"):
    """
    Create and save a graph of task vs. accuracy.

    Args:
        task_numbers (list of int): Task indices (x-axis).
        accuracies (list of float): Corresponding accuracies (y-axis).
        output_path (str): Path to save the graph.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(task_numbers, accuracies, marker='o', linestyle='-', linewidth=2, label="Accuracy")
    plt.title("Task vs Accuracy", fontsize=14)
    plt.xlabel("Task", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True)
    plt.xticks(task_numbers)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Graph saved at {output_path}")