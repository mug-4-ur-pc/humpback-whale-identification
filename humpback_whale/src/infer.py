from pathlib import Path
from typing import Optional, Union

import fire
import numpy as np
import onnxruntime as ort
import pandas as pd
import PIL
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from humpback_whale.src.data.transforms import create_infer_transforms
from humpback_whale.src.model.encoder import LabelEncoder


class InferenceModel:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        model_path = Path(cfg.model_path)

        self.transforms = create_infer_transforms(cfg.image_size, cfg.transforms)
        self.encoder = LabelEncoder.load(cfg.unique_labels)
        self.model_type = model_path.suffix[1:]

        if self.model_type == "pt":
            self._load_from_pt(model_path)
        elif self.model_type == "onnx":
            self._load_from_onnx(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")

    def _load_from_pt(self, model_path: Path):
        model_data = torch.load(model_path, weights_only=False, map_location="cpu")

        if isinstance(model_data, torch.jit.ScriptModule) or isinstance(
            model_data, torch.nn.Module
        ):
            self.model = model_data
        else:
            raise ValueError("Unsupported PyTorch checkpoint format")

        self.model.eval()
        print(f"Loaded PyTorch model: {model_path}")

    def _load_from_onnx(self, model_path: Path):
        self.sess = ort.InferenceSession(str(model_path))
        self.input_name = self.sess.get_inputs()[0].name
        print(f"Loaded ONNX model: {model_path}")

    def _preprocess_image(self, img: PIL.Image) -> Union[torch.Tensor, np.ndarray]:
        img = img.convert("RGB")
        img = self.transforms(img).unsqueeze(0)
        return img

    def _get_labels(self, img: torch.tensor) -> np.ndarray:
        if self.model_type == "pt":
            with torch.no_grad():
                logits = self.model(img).numpy()
        else:
            logits = self.sess.run(None, {self.input_name: img.numpy()})[0]

        outputs = np.argsort(logits[0])[::-1][: self.cfg.num_predictions]
        return outputs

    def predict(self, img: PIL.Image) -> list[str]:
        img = self._preprocess_image(img)
        label_ids = self._get_labels(img)
        labels = [self.encoder.decode(label_id) for label_id in label_ids]

        return labels

    def process_image(self, img_path: Path):
        try:
            img = PIL.Image.open(img_path)
            return self.predict(img)
        except PIL.UnidentifiedImageError as e:
            print(e)
            return None

    def _save_result(
        self, dir_paths: str, dir_labels: list[str], output_path: str | Path
    ):
        df = pd.DataFrame({"Image": dir_paths, "Id": dir_labels})
        df.to_csv(output_path)

    def process_directory(self, input_dir: str | Path, output_path: str | Path):
        input_dir = Path(input_dir)

        dir_paths, dir_labels = [], []
        for img_path in tqdm(input_dir.iterdir()):
            labels = self.process_image(img_path)
            dir_paths.append(img_path.name)
            dir_labels.append(labels)

        self._save_result(dir_paths, dir_labels, output_path)
        print()
        print(f"Results were saved to {output_path}")


def infer(
    input_path: str,
    output_path: str = "predictions.csv",
    config_path: Optional[str] = "config/infer.yaml",
):
    """
    Inference.

    Args:
        input_path: Path to image folder or one image
        output_path: Output YAML file path if image folder was given (default: predictions.csv)
        config_path: Path to YAML configuration file for inference (optional)
    """
    cfg = OmegaConf.load(config_path)
    input_path = Path(input_path)
    model = InferenceModel(cfg)

    if input_path.is_dir():
        model.process_directory(input_path, output_path)
    else:
        labels = model.process_image(input_path)
        print(f"Predicted whales: {labels}")


if __name__ == "__main__":
    fire.Fire(infer)
