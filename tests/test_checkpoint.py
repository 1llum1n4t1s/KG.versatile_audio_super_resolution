import ast
from pathlib import Path

import torch

import audiosr.checkpoint as checkpoint


ROOT = Path(__file__).parents[1]


def test_load_checkpoint_uses_tensor_only_torch_loader(monkeypatch):
    calls = []
    expected = {"state_dict": {"weight": torch.ones(1)}}

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return expected

    monkeypatch.setattr(checkpoint.torch, "load", fake_load)

    assert checkpoint.load_checkpoint("model.ckpt") is expected
    assert calls == [
        ("model.ckpt", {"map_location": "cpu", "weights_only": True})
    ]


def test_load_checkpoint_uses_safetensors_loader(monkeypatch):
    calls = []
    expected = {"weight": torch.ones(1)}
    monkeypatch.setattr(
        checkpoint,
        "load_file",
        lambda path, device: calls.append((path, device)) or expected,
    )
    monkeypatch.setattr(
        checkpoint.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    assert checkpoint.load_checkpoint("model.safetensors", torch.device("cpu")) is expected
    assert calls == [("model.safetensors", "cpu")]


def test_production_torch_loads_are_explicitly_tensor_only():
    unsafe_calls = []

    def dotted_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    for path in (ROOT / "audiosr").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if dotted_name(node.func) not in {
                "torch.load",
                "torch.hub.load_state_dict_from_url",
            }:
                continue
            weights_only = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "weights_only"),
                None,
            )
            if not isinstance(weights_only, ast.Constant) or weights_only.value is not True:
                unsafe_calls.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert unsafe_calls == []
