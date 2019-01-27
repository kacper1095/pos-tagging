from typing import *
from steps import Step


class Pipeline:
    def __init__(self, steps: List[Step]):
        self.steps = steps

    def evaluate(self):
        for i in range(1, len(self.steps)):
            self.steps[i](self.steps[i - 1])
        return self.steps[-1].evaluate()

    def get_result(self):
        return self.steps[-1].outputs
