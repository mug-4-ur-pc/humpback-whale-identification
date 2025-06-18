from pathlib import Path

import onnx
import tensorrt as trt
import torch
from omegaconf import DictConfig

from humpback_whale.src.model.identifier import IdentificationModel


class ModelExporter:
    """
    Export PyTorch .ckpt models to various formats.
    Supported formats: .pt, .onnx, .engine (TensorRT)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = trt.Logger(trt.Logger.WARNING)

    def to_pt(
        self,
        ckpt_path: str,
        output_path: str = None,
    ):
        """
        Export .ckpt to .pt format
        Args:
            ckpt_path: Path to input .ckpt file
            output_path: Optional output path (defaults to same directory as input)
        """
        ckpt_path = Path(ckpt_path)
        if not output_path:
            output_path = ckpt_path.with_suffix(".pt")
        checkpoint = torch.load(ckpt_path, weights_only=False)

        model = IdentificationModel(self.cfg.model, self.cfg.train)
        model.load_state_dict(checkpoint["state_dict"])

        torch.save(model, output_path)

        print(f"Successfully exported to {output_path}")

    def to_onnx(
        self,
        ckpt_path: str,
        output_path: str = None,
        input_names: list = ["input"],
        output_names: list = ["output"],
        dynamic_axes: dict = None,
        opset_version: int = 11,
    ):
        """
        Export .ckpt to ONNX format
        Args:
            ckpt_path: Path to input .ckpt file
            output_path: Optional output path
            input_names: List of input names
            output_names: List of output names
            dynamic_axes: Dict specifying dynamic axes
            opset_version: ONNX opset version
        """

        pt_path = Path(ckpt_path).with_suffix(".pt")
        if not pt_path.exists():
            self.to_pt(ckpt_path, pt_path)

        ckpt_path = Path(ckpt_path)
        if not output_path:
            output_path = ckpt_path.with_suffix(".onnx")

        model = torch.load(pt_path, weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Model in .ckpt file must be a PyTorch Module for ONNX export"
            )
        model.eval()

        input_shape = (1, 3, self.cfg.model.image_size, self.cfg.model.image_size)

        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        print(f"Successfully exported to {output_path}")

    def to_engine(
        self,
        ckpt_path: str,
        output_path: str = None,
        fp16_mode: bool = False,
    ):
        """
        Export .ckpt to TensorRT engine format
        Args:
            ckpt_path: Path to input .ckpt file
            output_path: Optional output path
            fp16_mode: Enable FP16 precision
        """
        onnx_path = Path(ckpt_path).with_suffix(".onnx")
        if not onnx_path.exists():
            self.to_onnx(ckpt_path, onnx_path)

        if not output_path:
            output_path = Path(ckpt_path).with_suffix(".engine")

        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError("ONNX parser failed")

        config = builder.create_builder_config()
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_serialized_network(network, config)
        with open(output_path, "wb") as f:
            f.write(engine)

        print(f"Successfully exported to {output_path}")
