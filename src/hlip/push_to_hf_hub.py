#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath('..'))

import json
import shutil
import argparse
from pathlib import Path
import importlib.util

import torch
import safetensors.torch

# HF Hub
from huggingface_hub import create_repo, upload_folder

# OpenCLIP
from open_clip.factory import _MODEL_CONFIGS
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.push_to_hf_hub import save_for_hf  # writes HF config + tokenizer

# Ensure your custom code is importable locally
import hlip                      # namespace package is fine
import hlip.visual_encoder       # important: registers your visual encoder


def load_checkpoint(path: str):
    """Robust torch.load that uses weights_only when available."""
    try:
        # torch>=2.4 supports weights_only; older torch will TypeError
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_module_prefix(sd: dict):
    """Remove 'module.' prefix from DDP state dict keys."""
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def find_package_paths(pkg_name: str):
    """
    Return all source directories for a package, including namespace packages.
    We will merge-copy them into the staging dir.
    """
    paths = []
    spec = importlib.util.find_spec(pkg_name)
    if spec and spec.submodule_search_locations:
        paths.extend([Path(p).resolve() for p in spec.submodule_search_locations])

    # Fallback scan of sys.path for a folder named pkg_name
    if not paths:
        for p in map(Path, sys.path):
            cand = (p / pkg_name)
            if cand.is_dir():
                paths.append(cand.resolve())

    if not paths:
        raise RuntimeError(f"Could not locate package '{pkg_name}' on disk.")
    # Deduplicate
    uniq = []
    seen = set()
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def vendor_namespace_package(pkg_name: str, dest_dir: Path):
    """Copy a (possibly namespace) package into dest_dir/pkg_name."""
    src_paths = find_package_paths(pkg_name)
    pkg_dst = dest_dir / pkg_name
    pkg_dst.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.so", ".git", ".pytest_cache")
    for src in src_paths:
        shutil.copytree(src, pkg_dst, dirs_exist_ok=True, ignore=ignore)
    # Convert to regular package in the snapshot for simpler downstream imports
    (pkg_dst / "__init__.py").touch(exist_ok=True)
    return pkg_dst


def main():
    parser = argparse.ArgumentParser("Push custom OpenCLIP model + code to HF Hub")
    parser.add_argument("--repo-id", type=str,
                        default="Zch0414/brainmri_clip_vit_base_multiscan_h2_token1176",
                        help="HF Hub repo id like user_or_org/name")
    parser.add_argument("--model-name", type=str,
                        default="clip_vit_base_multiscan_h2_token1176",
                        help="Must match your JSON config filename without .json")
    parser.add_argument("--ckpt", type=str, required=False,
                        default="/nfs/turbo/umms-tocho-snr/exp/chuizhao/hlip/release/brainmri_clip_vit_base_multiscan_h2_token1176.pt",
                        help="Path to your trained checkpoint .pt")
    parser.add_argument("--model-config-dir", type=str, default="./model_configs",
                        help="Folder containing your OpenCLIP JSON configs")
    parser.add_argument("--private", action="store_true", help="Create/keep repo private")
    parser.add_argument("--commit-message", type=str,
                        default="Add weights (safetensors+bin), config, tokenizer, and custom hlip code.",
                        help="Commit message for the upload")
    args = parser.parse_args()

    repo_id = args.repo_id
    model_name = args.model_name
    ckpt_path = Path(args.ckpt)
    model_cfg_dir = Path(args.model_config_dir)

    assert model_cfg_dir.is_dir(), f"Missing model config dir: {model_cfg_dir}"
    assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"

    print(f"[1/7] Ensuring repo exists: {repo_id}")
    create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"[2/7] Registering model configs from {model_cfg_dir}")
    for p in model_cfg_dir.glob("*.json"):
        with p.open("r") as f:
            _MODEL_CONFIGS[p.stem] = json.load(f)

    print(f"[3/7] Building model: {model_name}")
    model, _, _ = create_model_and_transforms(model_name, device="cpu", output_dict=True)

    print(f"[4/7] Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(str(ckpt_path))
    sd = ckpt.get("state_dict", ckpt)
    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Warning: missing keys: {len(missing)} (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"  Warning: unexpected keys: {len(unexpected)} (showing first 5): {unexpected[:5]}")

    print(f"[5/7] Preparing tokenizer and model_config")
    tokenizer = get_tokenizer(model_name)
    with (model_cfg_dir / f"{model_name}.json").open("r") as f:
        model_config = json.load(f)
    model_config.setdefault("text_cfg", {})
    model_config["text_cfg"]["hf_tokenizer_name"] = repo_id

    tmpdir = Path("../../hf_staging")
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)
    print(f"[6/7] Staging to {tmpdir}")

    # 6a) Write HF config + tokenizer, but skip weights; we will save contiguous copies
    save_for_hf(
        model,
        tokenizer=tokenizer,
        model_config=model_config,
        save_directory=str(tmpdir),
        safe_serialization="both",
        skip_weights=True,
    )

    # 6b) Save contiguous weights
    print("      Saving contiguous weights (.safetensors and .bin)")
    sd_contig = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    safetensors_path = tmpdir / "model.safetensors"
    bin_path = tmpdir / "pytorch_model.bin"
    safetensors.torch.save_file(sd_contig, safetensors_path)
    torch.save(sd_contig, bin_path)

    # 6c) Vendor your custom code (hlip/) and model_configs/
    print("      Vendoring custom package: hlip/")
    vendor_namespace_package("hlip", tmpdir)

    print("      Copying model_configs/")
    shutil.copytree(model_cfg_dir, tmpdir / "model_configs", dirs_exist_ok=True)

    # 6d) Optional: requirements for consumers
    (tmpdir / "requirements.txt").write_text(
        "open-clip-torch\n"
        "torch\n"
        "timm\n"
    )

    print(f"[7/7] Uploading folder to Hub: {repo_id}")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(tmpdir),
        commit_message=args.commit_message,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
