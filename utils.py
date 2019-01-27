import datetime
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

import common


def get_time_stamp() -> str:
    return datetime.datetime.now().strftime("%d_%m_%H_%M")


def evaluate_final_submission(all_predictions: list, path: Path):
    all_predictions = np.asarray(all_predictions, dtype=np.float32)
    all_predictions = np.mean(all_predictions, axis=0)
    all_predictions = np.argmax(all_predictions, axis=-1)
    evaluate_single_submission(all_predictions, path)


def evaluate_single_submission(predictions: Union[np.ndarray, list], path: Path):
    ids = np.arange(0, len(predictions))
    frame = pd.DataFrame(data={'id': ids + 1, 'labels': predictions})
    frame.to_csv(path, index=False)


def fscore(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    indices = np.where(y_true != common.PAD_TOKEN)[0]
    y_pred = np.argmax(y_pred[indices], axis=-1)
    y_true = y_true[indices]
    return f1_score(y_true, y_pred, average="weighted")


def acc(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    indices = np.where(y_true != common.PAD_TOKEN)[0]
    y_pred = np.argmax(y_pred[indices], axis=-1)
    y_true = y_true[indices]
    return accuracy_score(y_true, y_pred)


def get_ground_truth_from_dataset(dataset: DataLoader):
    gt = []
    for _, y, lengths, masks in dataset:
        for sample, length in zip(y, lengths):
            gt.extend(sample[:length].cpu().numpy().flatten().tolist())
    return gt
