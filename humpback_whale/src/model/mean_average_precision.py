import numpy as np
import torch
import torchmetrics as tm


class FixedRetrievalMAP(torch.nn.Module):
    """
    This is RetrievalMAP implementation with memory leak fix
    """

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.rmap = []

    def forward(self, preds, labels):
        metrics = [
            tm.retrieval.average_precision.retrieval_average_precision(
                p, l, top_k=self.top_k
            ).item()
            for p, l, in zip(preds, labels)
        ]
        self.rmap.extend(metrics)
        return np.mean(metrics)

    def update(self, preds, labels):
        self.forward(preds, labels)

    def compute(self):
        result = np.mean(self.rmap)
        self.rmap.clear()
        return result
