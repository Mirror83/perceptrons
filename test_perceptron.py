from enum import Enum, auto
from perceptron import Perceptron

import numpy as np


class LogicOp(Enum):
    AND = auto()
    OR = auto()


def logic_inputs_targets(op: LogicOp):
    inputs = inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]]
    )

    targets = np.zeros(inputs.shape[0])
    for i, x in enumerate(inputs):
        match op:
            case LogicOp.AND:
                targets[i] = x[1] & x[2]
            case LogicOp.OR:
                targets[i] = x[1] | x[2]

    return inputs, targets


class TestPerceptron:
    def test_and(self):
        inputs, targets = logic_inputs_targets(LogicOp.AND)
        p = Perceptron(inputs, targets).train()
        assert np.all(p.predict() == targets)

    def test_or(self):
        inputs, targets = logic_inputs_targets(LogicOp.OR)
        p = Perceptron(inputs, targets).train()
        assert np.all(p.predict() == targets)
