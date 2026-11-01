"""U-Net attention用LoRA。ベースのcheckpointとは別のsafetensorsへ保存する。"""

import copy
import hashlib
import json
import math
import os

import torch
from torch import nn
from torch.nn import functional as F


_FORMAT = "audiosr-lora-v1"
_CONFIG_ATTR = "_audiosr_lora_config"
_BASE_ATTR = "_audiosr_base_metadata"


class LoRALinear(nn.Module):
    """元のLinearを保持し、低rankの差分だけを学習する。"""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.train(base.training)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        residual = F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B)
        return self.base(value) + residual * (self.alpha / self.rank)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_hyperparameters(rank, alpha, dropout):
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("LoRA alpha must be finite and positive")
    try:
        valid_alpha = math.isfinite(alpha) and alpha > 0
    except OverflowError:
        valid_alpha = False
    if not valid_alpha:
        raise ValueError("LoRA alpha must be finite and positive")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("LoRA dropout must be finite and in [0, 1)")
    try:
        valid_dropout = math.isfinite(dropout) and 0 <= dropout < 1
    except OverflowError:
        valid_dropout = False
    if not valid_dropout:
        raise ValueError("LoRA dropout must be finite and in [0, 1)")


def _unet(model):
    unet = getattr(getattr(model, "model", None), "diffusion_model", None)
    if not isinstance(unet, nn.Module):
        raise ValueError("Expected model.model.diffusion_model to be an nn.Module")
    return unet


def _ensure_unmodified(model):
    if hasattr(model, _CONFIG_ATTR) or any(
        isinstance(module, LoRALinear) for module in model.modules()
    ):
        raise ValueError("A LoRA adapter is already attached to this model")


def _targets(unet):
    targets = {}
    for name, module in unet.named_modules():
        if type(module).__name__ != "CrossAttention":
            continue
        for suffix in ("to_q", "to_k", "to_v", "to_out.0"):
            linear = module.get_submodule(suffix)
            if not isinstance(linear, nn.Linear):
                raise ValueError(f"Unsupported LoRA target: {name}.{suffix}")
            if not linear.weight.is_floating_point() or linear.weight.device.type == "meta":
                raise ValueError(f"LoRA requires materialized floating point weights: {name}.{suffix}")
            target = f"{name}.{suffix}" if name else suffix
            targets[target] = linear
    if not targets:
        raise ValueError("No CrossAttention Linear layers found in the U-Net")
    if len({id(module) for module in targets.values()}) != len(targets):
        raise ValueError("Shared Linear targets are unsupported")
    return targets


def _signature(unet):
    # shapeだけでは検出できないattention設定と、層構造も指紋に含める。
    structure = []
    for name, module in unet.named_modules():
        properties = {}
        for key in ("heads", "scale", "in_channels", "out_channels", "model_channels",
                    "num_res_blocks", "attention_resolutions", "channel_mult",
                    "num_heads", "num_head_channels", "num_heads_upsample",
                    "num_classes", "extra_film_condition_dim", "predict_codebook_ids"):
            if hasattr(module, key):
                value = getattr(module, key)
                if value is None or isinstance(value, (bool, int, float, str, list, tuple)):
                    properties[key] = value
        structure.append([name, f"{type(module).__module__}.{type(module).__qualname__}", properties])
    shapes = [[name, list(tensor.shape)] for name, tensor in unet.state_dict().items()]
    return hashlib.sha256(_json([structure, shapes]).encode("utf-8")).hexdigest()


def _base(model):
    base = getattr(model, _BASE_ATTR, {})
    if not isinstance(base, dict):
        raise ValueError(f"{_BASE_ATTR} must be a JSON object")
    _json(base)
    return copy.deepcopy(base)


def _configuration(model, targets, rank, alpha, dropout):
    if any(alpha / rank > torch.finfo(layer.weight.dtype).max for layer in targets.values()):
        raise ValueError("LoRA scaling exceeds the base model dtype range")
    return {
        "format": _FORMAT,
        "rank": rank,
        "alpha": float(alpha),
        "dropout": float(dropout),
        "targets": {
            name: {"in_features": layer.in_features, "out_features": layer.out_features,
                   "bias": layer.bias is not None}
            for name, layer in targets.items()
        },
        "unet_signature": _signature(_unet(model)),
        "base": _base(model),
    }


def _install(model, targets, config, weights=None):
    # 全wrapperの構築と重み検証を済ませてから、モデルの構造を変更する。
    replacements = {}
    for name, base in targets.items():
        layer = LoRALinear(base, config["rank"], config["alpha"], config["dropout"])
        if weights is not None:
            with torch.no_grad():
                layer.lora_A.copy_(weights[f"{name}.lora_A"])
                layer.lora_B.copy_(weights[f"{name}.lora_B"])
        replacements[name] = layer
    model.requires_grad_(False)
    unet = _unet(model)
    for name, layer in replacements.items():
        parent_name, _, attribute = name.rpartition(".")
        parent = unet.get_submodule(parent_name) if parent_name else unet
        setattr(parent, attribute, layer)
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
    setattr(model, _CONFIG_ATTR, copy.deepcopy(config))


def inject_lora(model, rank=8, alpha=8.0, dropout=0.0):
    """ベース全体をfreezeし、U-NetのattentionへLoRAを注入する。"""
    _validate_hyperparameters(rank, alpha, dropout)
    _ensure_unmodified(model)
    targets = _targets(_unet(model))
    config = _configuration(model, targets, rank, alpha, dropout)
    _install(model, targets, config)
    return copy.deepcopy(config)


def lora_parameters(model):
    """Optimizerへ渡すLoRAパラメーターを返す。"""
    for module in _unet(model).modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def save_lora(model, path, metadata=None):
    """LoRA差分とJSON metadataを一つのsafetensorsへ保存する。"""
    from safetensors.torch import save_file

    if not os.fspath(path).lower().endswith(".safetensors"):
        raise ValueError("LoRA adapters must use the .safetensors extension")
    config = getattr(model, _CONFIG_ATTR, None)
    if not isinstance(config, dict):
        raise ValueError("No LoRA adapter is attached to this model")
    if _base(model) != config["base"]:
        raise ValueError("Base metadata changed after LoRA injection")
    metadata = {} if metadata is None else copy.deepcopy(metadata)
    if not isinstance(metadata, dict) or "lora" in metadata:
        raise ValueError("metadata must be a JSON object without the reserved 'lora' key")
    if "base" in metadata and metadata["base"] != config["base"]:
        raise ValueError("metadata base does not match the attached adapter")
    metadata["lora"] = copy.deepcopy(config)
    encoded = _json(metadata)
    weights = {}
    unet = _unet(model)
    for name in config["targets"]:
        layer = unet.get_submodule(name)
        if not isinstance(layer, LoRALinear):
            raise ValueError(f"LoRA target was replaced: {name}")
        for key in ("lora_A", "lora_B"):
            tensor = getattr(layer, key).detach().cpu().contiguous()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite LoRA weights: {name}.{key}")
            weights[f"{name}.{key}"] = tensor
    save_file(weights, os.fspath(path), metadata={"audiosr": encoded})
    return metadata


def load_lora(model, path, expected_base=None):
    """全適合性と全tensorを検証し、合格後だけadapterを適用する。"""
    from safetensors import safe_open

    if not os.fspath(path).lower().endswith(".safetensors"):
        raise ValueError("LoRA adapters must use the .safetensors extension")
    _ensure_unmodified(model)
    with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
        try:
            metadata = json.loads((handle.metadata() or {})["audiosr"])
            config = metadata["lora"]
            if not isinstance(metadata, dict) or not isinstance(config, dict):
                raise ValueError("Expected JSON objects")
            _validate_hyperparameters(config["rank"], config["alpha"], config["dropout"])
            _json(metadata)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid AudioSR LoRA metadata") from error
        targets = _targets(_unet(model))
        current = _configuration(model, targets, config["rank"], config["alpha"], config["dropout"])
        if config != current:
            raise ValueError("LoRA adapter does not match the base model, U-Net, or target configuration")
        if expected_base is not None:
            if not isinstance(expected_base, dict) or any(
                key not in config["base"] or config["base"][key] != value
                for key, value in expected_base.items()
            ):
                raise ValueError("LoRA adapter does not match expected_base")
        shapes = {}
        for name, layer in targets.items():
            shapes[f"{name}.lora_A"] = (config["rank"], layer.in_features)
            shapes[f"{name}.lora_B"] = (layer.out_features, config["rank"])
        if set(handle.keys()) != set(shapes):
            raise ValueError("LoRA tensor names do not match the declared targets")
        weights = {}
        for name, shape in shapes.items():
            tensor = handle.get_tensor(name)
            if tuple(tensor.shape) != shape or not tensor.is_floating_point():
                raise ValueError(f"Invalid LoRA tensor shape or dtype: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite LoRA weights: {name}")
            target_name = name.rsplit(".", 1)[0]
            converted = tensor.to(dtype=targets[target_name].weight.dtype)
            if not torch.isfinite(converted).all():
                raise ValueError(f"LoRA weights overflow base dtype: {name}")
            weights[name] = converted
    _install(model, targets, config, weights)
    return copy.deepcopy(metadata)
