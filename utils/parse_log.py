import os
import matplotlib.pyplot as plt
from collections import defaultdict


LOG_DIR = os.path.join('../log/20.07.18_18-32')
SAVE = True


def get_figures(path_to_log=os.path.join(LOG_DIR, 'logger.log')):
    val_losses = []
    train_losses = []
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)

    with open(path_to_log, 'r') as log_file:
        while True:
            line = log_file.readline()
            if not line:
                break

            line = line.strip()
            if line.find('Train loss') != -1:
                train_losses.append(float(line.split(':')[1]))

            elif line.find('Val loss') != -1:
                val_losses.append(float(line.split(':')[1]))

            elif line.find('Train metrics') != -1:
                metrics = line.split('---> ')[1].split('\t')
                for metric in metrics:
                    m_name, m_value = metric.split(':')
                    train_metrics[m_name.strip()].append(float(m_value))

            elif line.find('Val metrics') != -1:
                metrics = line.split('---> ')[1].split('\t')
                for metric in metrics:
                    m_name, m_value = metric.split(':')
                    val_metrics[m_name.strip()].append(float(m_value))

    return train_losses, val_losses, train_metrics, val_metrics


if __name__ == '__main__':
    train_losses, val_losses, train_metrics, val_metrics = get_figures()
    plt.figure(figsize=(12, 8))
    plt.grid()
    plt.plot(train_losses, '-bo', label='Train')
    plt.plot(val_losses, '--ro', label='Val')
    plt.title('BCE loss', fontsize=17)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss value', fontsize=14)
    plt.legend(prop={'size': 17})
    if SAVE:
        plt.savefig(os.path.join(LOG_DIR, 'losses.png'))
    plt.show()

    for m_name in train_metrics:
        plt.figure(figsize=(12, 8))
        plt.grid()
        plt.plot(train_metrics[m_name], '-bo', label='Train')
        plt.plot(val_metrics[m_name], '--ro', label='Val')
        plt.title(f'{m_name}', fontsize=17)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel(f'{m_name} value', fontsize=14)
        plt.legend(prop={'size': 17})
        if SAVE:
            plt.savefig(os.path.join(LOG_DIR, f'{m_name}.png'))
        plt.show()




