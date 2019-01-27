from typing import *
from pathlib import Path

import fastText
import pickle as pkl
import numpy as np


class Step:
    def __init__(self, name: str):
        self.inputs: List["Step"] = []
        self.outputs: List = []
        self.name = name

    def __call__(self, *input_steps: "Step"):
        self.inputs.extend(input_steps)

    def evaluate(self):
        for step in self.inputs:
            step.evaluate()
        self.transform()
        return self.outputs

    def transform(self, **kwargs):
        raise NotImplementedError


class Load(Step):
    def __init__(self, name: str, path: str):
        super().__init__(name)
        self.path = path

    def transform(self, **kwargs):
        self.outputs.append(np.load(self.path))


class ToLower(Step):
    def transform(self, **kwargs):
        data = self.inputs[0].outputs[0]
        output = []
        for seq in data:
            output.append([])
            for word in seq:
                output[-1].append(word.lower())
        self.outputs = np.asarray(output)


class ToEmbedding(Step):
    def __init__(self, name: str, model_path: str, embeddings_path: str):
        super().__init__(name)
        self.model_path = model_path
        self.embeddings_path = embeddings_path

    def transform(self, **kwargs):
        if Path(self.embeddings_path).exists():
            with open(self.embeddings_path, 'rb') as f:
                self.outputs = pkl.load(f)
            return
        embedding_model = fastText.load_model(self.model_path)
        inputs = self.inputs[0].outputs
        output = []
        for seq in inputs:
            output.append([])
            for word in seq:
                output[-1].append(embedding_model.get_word_vector(word))
        self.outputs = output
        with open(self.embeddings_path, 'wb') as f:
            pkl.dump(self.outputs, f, protocol=pkl.HIGHEST_PROTOCOL)
        del embedding_model
