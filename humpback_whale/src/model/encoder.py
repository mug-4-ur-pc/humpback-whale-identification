import json
from collections import defaultdict
from typing import Dict, List

import torch


class LabelEncoder:
    def __init__(self, label_to_idx: Dict[str, int], unknown_label: str):
        """
        Initialize the encoder with a label-to-index mapping.

        Args:
            label_to_idx: Dictionary mapping labels to integer indices
            unknown_label: Label to return for unknown indices during decoding
        """
        self.label_to_idx = label_to_idx
        self.unknown_label = unknown_label
        self.unknown_idx = label_to_idx.get(unknown_label, len(label_to_idx))

        self.idx_to_label = defaultdict(lambda: unknown_label)
        self.idx_to_label.update({idx: label for label, idx in label_to_idx.items()})

    @staticmethod
    def create(labels: List[str], unknown_label: str) -> "LabelEncoder":
        """
        Create a LabelEncoder from a list of labels.

        Args:
            labels: List of labels (can contain duplicates)

        Returns:
            LabelEncoder instance
        """
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return LabelEncoder(label_to_idx, unknown_label)

    def encode(
        self, labels: str | List[str], as_tensor: bool = False
    ) -> int | List[int] | torch.Tensor:
        """
        Encode label(s) to integer indices.

        Args:
            labels: Single label or list of labels
            as_tensor: Return as PyTorch tensor

        Returns:
            Single index or list of indices
        """
        if isinstance(labels, str):
            return self.label_to_idx.get(labels, self.unknown_idx)

        encoded = [self.label_to_idx.get(label, self.unknown_idx) for label in labels]
        return torch.tensor(encoded, dtype=torch.long) if as_tensor else encoded

    def decode(self, indices: int | List[int] | torch.Tensor) -> str | List[str]:
        """
        Decode integer index/indices back to label(s).

        Args:
            indices: Single index or list/array of indices

        Returns:
            Single label or list of labels
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        if isinstance(indices, list):
            return [self.idx_to_label[idx] for idx in indices]
        return self.idx_to_label[indices]

    def save(self, file_path: str):
        """Save the encoder to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(
                {
                    "label_to_idx": self.label_to_idx,
                    "unknown_label": self.unknown_label,
                },
                f,
            )

    @classmethod
    def load(cls, file_path: str) -> "LabelEncoder":
        """Load an encoder from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(
            label_to_idx=data["label_to_idx"], unknown_label=data["unknown_label"]
        )

    @property
    def classes(self) -> List[str]:
        """Get the list of classes in order."""
        return [self.idx_to_label[idx] for idx in range(len(self.label_to_idx))]

    def __len__(self) -> int:
        """Return the number of unique classes."""
        return len(self.label_to_idx)

    def __contains__(self, label: str) -> bool:
        """Check if label exists in the encoder."""
        return label in self.label_to_idx
