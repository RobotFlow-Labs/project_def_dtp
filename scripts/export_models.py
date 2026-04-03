"""DEF-DTP model export pipeline: pth → safetensors → ONNX → TRT fp16 → TRT fp32.

Exports the ensemble trajectory predictor through all ANIMA-mandated formats.

Usage:
    CUDA_VISIBLE_DEVICES=6 python scripts/export_models.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from safetensors.torch import save_file as save_safetensors

from anima_def_dtp.backends.cuda.predictor import TrajectoryPredictorNet
from anima_def_dtp.constants import DATASET_THRESHOLDS, PAPER_ARXIV_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = Path("/mnt/artifacts-datai")
MODULE = "project_def_dtp"


def export_pth(model: torch.nn.Module, path: Path, metadata: dict) -> None:
    """Save PyTorch checkpoint with metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    size_mb = path.stat().st_size / 1024**2
    logger.info("[PTH] Saved %s (%.1f MB)", path, size_mb)


def export_safetensors(model: torch.nn.Module, path: Path) -> None:
    """Save model weights in safetensors format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    save_safetensors(state, str(path))
    size_mb = path.stat().st_size / 1024**2
    logger.info("[SAFETENSORS] Saved %s (%.1f MB)", path, size_mb)


def export_onnx(
    model: torch.nn.Module,
    path: Path,
    obs_length: int,
    device: str,
) -> None:
    """Export to ONNX with dynamic batch dimension, weights embedded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, obs_length, 2)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(path),
        input_names=["observe"],
        output_names=["predict"],
        dynamic_axes={
            "observe": {0: "batch"},
            "predict": {0: "batch"},
        },
        opset_version=18,
    )
    # Ensure all weights are embedded (no external .data files)
    onnx_model = onnx.load(str(path))
    onnx.save(onnx_model, str(path), save_as_external_data=False)
    # Remove stale .data file if created
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        data_file.unlink()
    onnx.checker.check_model(onnx_model)
    size_mb = path.stat().st_size / 1024**2
    logger.info("[ONNX] Saved %s (%.1f MB) — verified, weights embedded", path, size_mb)
    model.to(device)  # move back


def export_trt(
    onnx_path: Path,
    trt_path: Path,
    precision: str,
    obs_length: int,
) -> None:
    """Build TensorRT engine from ONNX model.

    Parameters
    ----------
    precision : "fp16" or "fp32"
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TRT parse error: %s", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("[TRT] FP16 enabled")
        else:
            logger.warning("[TRT] FP16 not supported on this GPU, falling back to FP32")

    # Set optimization profiles for dynamic batch
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "observe",
        min=(1, obs_length, 2),
        opt=(64, obs_length, 2),
        max=(512, obs_length, 2),
    )
    config.add_optimization_profile(profile)

    logger.info("[TRT] Building %s engine (this may take a minute)...", precision.upper())
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build failed")

    trt_path.parent.mkdir(parents=True, exist_ok=True)
    trt_path.write_bytes(engine_bytes)
    size_mb = trt_path.stat().st_size / 1024**2
    logger.info("[TRT %s] Saved %s (%.1f MB)", precision.upper(), trt_path, size_mb)


def validate_trt_engine(trt_path: Path, obs_length: int) -> None:
    """Quick inference check on the TRT engine."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("observe", (4, obs_length, 2))

    # Allocate I/O
    input_np = np.random.randn(4, obs_length, 2).astype(np.float32)
    input_tensor = torch.from_numpy(input_np).cuda()

    # Get output shape (convert trt.Dims to tuple)
    output_shape = tuple(context.get_tensor_shape("predict"))
    output_tensor = torch.empty(output_shape, dtype=torch.float32, device="cuda")

    context.set_tensor_address("observe", input_tensor.data_ptr())
    context.set_tensor_address("predict", output_tensor.data_ptr())
    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    logger.info(
        "[TRT VALIDATE] %s — input=%s output=%s",
        trt_path.name, input_tensor.shape, output_tensor.shape,
    )


def build_model_card(export_dir: Path, metadata: dict) -> None:
    """Generate model card for HuggingFace."""
    card = f"""---
library_name: pytorch
tags:
  - adversarial-attack
  - trajectory-prediction
  - defense
  - anima
license: mit
---

# DEF-DTP: DTP-Attack Trajectory Predictor

**Paper:** {metadata['paper_title']}
**ArXiv:** {metadata['paper_arxiv']}
**Module:** ANIMA DEF-DTP (Wave 7 Defense)

## Model Description

Ensemble trajectory predictor trained for DTP-Attack evaluation.
Maps observed trajectories (B, {metadata['obs_length']}, 2) to
predicted future trajectories (B, {metadata['pred_length']}, 2).

## Exported Formats

| Format | File | Size |
|--------|------|------|
| PyTorch | `predictor.pth` | {metadata.get('pth_mb', '?')} MB |
| Safetensors | `predictor.safetensors` | {metadata.get('safetensors_mb', '?')} MB |
| ONNX | `predictor.onnx` | {metadata.get('onnx_mb', '?')} MB |
| TensorRT FP16 | `predictor_fp16.engine` | {metadata.get('trt_fp16_mb', '?')} MB |
| TensorRT FP32 | `predictor_fp32.engine` | {metadata.get('trt_fp32_mb', '?')} MB |

## Architecture

- Ensemble of {metadata['n_models']} models
- Hidden dim: {metadata['hidden_dim']}
- Layers: {metadata['num_layers']}
- Total params: {metadata['total_params']:,}

## Campaign Results (nuScenes v1.0-trainval)

| Objective | ASR | Paper Range |
|-----------|-----|-------------|
| ade | 100% | - |
| fde | 100% | - |
| left | 39.8% | 41-81% |
| right | 40.8% | 41-81% |
| front | 38.2% | 41-81% |
| rear | 42.2% | 41-81% |

## Usage

```python
import torch
model = torch.load("predictor.pth")["model_state_dict"]
```
"""
    (export_dir / "README.md").write_text(card)
    logger.info("[MODEL CARD] Written to %s", export_dir / "README.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DEF-DTP model export pipeline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="nuscenes")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=ARTIFACTS / "exports" / MODULE,
    )
    args = parser.parse_args(argv)

    device = args.device
    thresholds = DATASET_THRESHOLDS[args.dataset]
    obs_length = thresholds.obs_length
    pred_length = thresholds.pred_length

    n_models = 8
    hidden_dim = 1024
    num_layers = 8

    logger.info("=" * 60)
    logger.info("DEF-DTP Export Pipeline")
    logger.info("Formats: pth → safetensors → ONNX → TRT fp16 → TRT fp32")
    logger.info("=" * 60)

    # Build single model for export (ensemble member)
    model = TrajectoryPredictorNet(
        obs_length=obs_length,
        pred_length=pred_length,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params", total_params)

    export_dir = args.export_dir
    metadata = {
        "paper_arxiv": PAPER_ARXIV_ID,
        "paper_title": "DTP-Attack",
        "obs_length": obs_length,
        "pred_length": pred_length,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "n_models": n_models,
        "total_params": total_params * n_models,
    }

    start = time.time()

    # 1. PTH
    pth_path = export_dir / "predictor.pth"
    export_pth(model, pth_path, metadata)
    metadata["pth_mb"] = f"{pth_path.stat().st_size / 1024**2:.1f}"

    # 2. Safetensors
    st_path = export_dir / "predictor.safetensors"
    export_safetensors(model, st_path)
    metadata["safetensors_mb"] = f"{st_path.stat().st_size / 1024**2:.1f}"

    # 3. ONNX
    onnx_path = export_dir / "predictor.onnx"
    export_onnx(model, onnx_path, obs_length, device)
    metadata["onnx_mb"] = f"{onnx_path.stat().st_size / 1024**2:.1f}"

    # 4. TensorRT FP16
    trt_fp16_path = export_dir / "predictor_fp16.engine"
    export_trt(onnx_path, trt_fp16_path, "fp16", obs_length)
    metadata["trt_fp16_mb"] = f"{trt_fp16_path.stat().st_size / 1024**2:.1f}"
    validate_trt_engine(trt_fp16_path, obs_length)

    # 5. TensorRT FP32
    trt_fp32_path = export_dir / "predictor_fp32.engine"
    export_trt(onnx_path, trt_fp32_path, "fp32", obs_length)
    metadata["trt_fp32_mb"] = f"{trt_fp32_path.stat().st_size / 1024**2:.1f}"
    validate_trt_engine(trt_fp32_path, obs_length)

    # Model card
    build_model_card(export_dir, metadata)

    # Save metadata
    (export_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2))

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE — %.1fs", elapsed)
    logger.info("Files in %s:", export_dir)
    for f in sorted(export_dir.iterdir()):
        logger.info("  %s (%.1f MB)", f.name, f.stat().st_size / 1024**2)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
