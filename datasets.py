from torch.utils.data import Dataset

import common
import numpy as np
from pipeline import *
from steps import *


def run_processing_data_pipeline(dataset_name: str, output_name: str):
    pipe = Pipeline([
        Load("load", dataset_name),
        ToLower("tolower"),
        ToEmbedding("toembedding", common.FASTTEXT_MODEL.as_posix(),
                    (common.PROCESSED_DATA / output_name).as_posix())
    ])
    return pipe.evaluate()


def load_data() -> (np.ndarray, np.ndarray):
    with open(common.PROCESSED_DATA / "embeddings.pkl", 'rb') as f:
        data = pkl.load(f)
    labels = np.load(common.RAW_DATA / "competition_dataset_labels.npy")
    return np.array(data, dtype=np.object), labels


def load_test_set() -> np.ndarray:
    with open(common.PROCESSED_DATA / "test_embeddings.pkl", 'rb') as f:
        data = pkl.load(f)
    return np.array(data, dtype=np.object)


def pad_values(sample: np.ndarray, value: int) -> np.ndarray:
    to_pad = common.LONGEST_SAMPLE - sample.shape[0]
    if len(sample.shape) == 2:
        padding = [[0, to_pad], [0, 0]]
    else:
        padding = [0, to_pad]
    return np.pad(sample, padding, mode='constant', constant_values=value)


class TrainDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = x_data
        self.y = y_data

    def __getitem__(self, index):
        sample = np.asarray(self.x[index], dtype=np.float32)
        y = np.asarray(self.y[index], dtype=np.float32)
        length = len(sample)

        sample = pad_values(sample, 0)

        y = pad_values(y, common.PAD_TOKEN).astype(np.int32)
        mask = np.zeros((common.LONGEST_SAMPLE,), dtype=np.float32)
        mask[:length] = 1
        return sample, y, length, mask

    def __len__(self):
        return len(self.x)


class ValidDataset(TrainDataset):
    pass


class TestDataset(Dataset):
    def __init__(self, x_data: np.ndarray):
        self.x = x_data

    def __getitem__(self, index):
        sample = np.asarray(self.x[index], dtype=np.float32)
        length = len(sample)

        sample = pad_values(sample, 0)
        y = np.zeros((len(sample),), dtype=np.float32)
        y = pad_values(y, common.PAD_TOKEN).astype(np.int32)
        mask = np.zeros((common.LONGEST_SAMPLE,), dtype=np.float32)
        mask[:length] = 1
        return sample, y, length, mask

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    run_processing_data_pipeline((common.RAW_DATA / "competition_dataset.npy"), "embeddings.pkl")
    # run_processing_data_pipeline((common.RAW_DATA / "competition_test.npy"), "test_embeddings.pkl")
