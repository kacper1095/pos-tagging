import numpy as np
import matplotlib.pyplot as plt
import common
import pandas as pd
import shutil
import logging
from main import run


def lr_test_run(epochs: int, batch_size: int, min_lr: float, max_lr: float, resolution: int):
    path = common.MODELS_PATH / 'lr_test'
    if path.exists():
        shutil.rmtree(path.as_posix())
    path.mkdir()

    metric_values = []
    learning_rates = np.linspace(min_lr, max_lr, resolution)
    for i, lr in enumerate(learning_rates):
        logging.info("Testing: {}. Lr: {} / {}".format(lr, i + 1, len(learning_rates)))
        last_val_metrics = run(None, epochs, batch_size, lr, kfolds=1, run_single_split=True)
        metric_values.append(last_val_metrics['fscore'])
    plt.figure()
    plt.plot(learning_rates, metric_values)
    plt.title("Learning rate test")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.savefig(path / 'lr_test.pdf')
    dataframe = pd.DataFrame(data={
        'lrs': learning_rates,
        'fscores': metric_values
    })
    dataframe.to_csv(path / 'lr_test.csv', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_lr", type=float, default=0.0001)
    parser.add_argument("--max_lr", type=float, default=1)
    parser.add_argument("--resolution", type=int, default=100)
    args = parser.parse_args()

    lr_test_run(args.epochs, args.batch_size, args.min_lr, args.max_lr, args.resolution)
