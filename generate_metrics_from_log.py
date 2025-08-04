import re
import matplotlib.pyplot as plt
from datetime import datetime


def parse_log_file(log_file_path):
    """Extracting data from training log which might named log.txt"""
    epochs = []
    steps = []
    losses = []
    steps_per_sec = []
    current_epoch = None
    timestamps = []

    # Match pattern (Time pattern looks weird, but it copied from log.txt and it works)
    epoch_pattern = r'Beginning epoch (\d+)'
    data_pattern = r'\(step=(\d+)\).*Train Loss: (\d+\.\d+).*Train Steps/Sec: (\d+\.\d+)'
    time_pattern = r'\[\[34m(.*?)\[0m\]'

    with open(log_file_path, 'r') as f:
        for line in f:
            # Match epoch
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            # Match timestamp (Maybe useful in somewhere?)
            time_match = re.search(time_pattern, line)
            if not time_match:
                continue
            timestamp = datetime.strptime(time_match.group(1), '%Y-%m-%d %H:%M:%S')

            # Match data
            data_match = re.search(data_pattern, line)
            if data_match:
                step = int(data_match.group(1))
                loss = float(data_match.group(2))
                speed = float(data_match.group(3))

                if current_epoch is not None:
                    epochs.append(current_epoch)
                    steps.append(step)
                    losses.append(loss)
                    steps_per_sec.append(speed)
                    timestamps.append(timestamp)

    return {
        'epochs': epochs,
        'steps': steps,
        'losses': losses,
        'steps_per_sec': steps_per_sec,
        'timestamps': timestamps
    }


def plot_training_metrics(data):
    # 1. Loss vs Steps
    plt.figure(figsize=(12, 6))
    plt.plot(data['steps'], data['losses'], 'b-')
    plt.title('Training Loss vs Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_vs_steps.png')
    plt.show()

    # 2. Loss vs Epochs
    plt.figure(figsize=(12, 6))

    # Compute loss/epoch
    unique_epochs = sorted(list(set(data['epochs'])))
    epoch_losses = []
    for epoch in unique_epochs:
        epoch_indices = [i for i, e in enumerate(data['epochs']) if e == epoch]
        epoch_loss = sum([data['losses'][i] for i in epoch_indices]) / len(epoch_indices)
        epoch_losses.append(epoch_loss)

    plt.plot(unique_epochs, epoch_losses, 'r-')
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_vs_epochs.png')
    plt.show()


if __name__ == "__main__":
    log_file_path = "log.txt"
    training_data = parse_log_file(log_file_path)
    plot_training_metrics(training_data)