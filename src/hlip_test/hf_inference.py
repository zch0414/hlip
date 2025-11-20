from pathlib import Path
import os, sys, json, torch, importlib

from huggingface_hub import snapshot_download
from open_clip.factory import _MODEL_CONFIGS
from open_clip import create_model_and_transforms, get_tokenizer, build_zero_shot_classifier

import safetensors.torch as st
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def loader(study_path, num_slices):
    imgs = []
    for scan in [os.path.join(study_path, p) for p in os.listdir(study_path)]:
        # load image tensor
        img = torch.load(scan, weights_only=True)
        if len(img.shape) == 4:
            img = img[:, :, :, 0]  # [C, H, W, 1] -> [C, H, W]
        img = img.float() / 255.0  # [C, H, W]
        _, h, w = img.shape

        # pad to square
        size = max(h, w)
        pad_h = size - h; pad_w = size - w
        left = pad_w // 2; right = pad_w - left; top = pad_h // 2; bottom = pad_h - top
        img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=0)

        # resize to 256, make depth=num_slices, center-crop to 224
        img = torch.nn.functional.interpolate(img[None, ...], size=(256, 256), mode='bilinear')[0]
        img = torch.nn.functional.interpolate(img[None, None, ...], size=(num_slices, 256, 256), mode='nearest-exact')[0, 0]
        img = img[:, 16:240, 16:240]  # [D, 224, 224]

        # normalize (scalar mean/std across slices-as-channels)
        normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(),
                               torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
        img = normalizer(img[None, ...])
        imgs.append(img)

    return torch.stack(imgs, dim=0)[None, ...]  # [1, n_scans, 1, D, H, W]


# ---- constants ----
REPO_ID = "Zch0414/hlip-2025-10-08"
MODEL_NAME = "ablate_seqposemb_vit_base_multiscan_h2_dinotxt1568"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------

# 1) download snapshot and make vendored package importable
repo_dir = Path(snapshot_download(repo_id=REPO_ID))
sys.path.append(str(repo_dir))
importlib.invalidate_caches()
print(f"[OK] repo_dir = {repo_dir}")

# 2) import your registry so timm knows the custom model name
import hlip.visual_encoder  # registers vit_base_multiscan_h2_token1176 with timm

# 3) load OpenCLIP config and register it
cfg = json.loads((repo_dir / "open_clip_config.json").read_text())
_MODEL_CONFIGS[MODEL_NAME] = cfg["model_cfg"]
print("[OK] registered MODEL_CONFIGS key:", MODEL_NAME)

# 4) build model and tokenizer
model, _, _ = create_model_and_transforms(
    MODEL_NAME, device=DEVICE, output_dict=True
)
tokenizer = get_tokenizer(MODEL_NAME)
print("[OK] model built on", DEVICE)
print("[OK] tokenizer ready")

# 5) load pretrained weights from the snapshot (prefers safetensors)
weight_path = None
for fname in ("model.safetensors", "pytorch_model.bin"):
    p = repo_dir / fname
    if p.exists():
        weight_path = p
        break
assert weight_path is not None, "No weights found in repo snapshot."

if weight_path.suffix == ".safetensors":
    state_dict = st.load_file(str(weight_path))
else:
    state_dict = torch.load(str(weight_path), map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"[OK] loaded weights: {weight_path.name} | missing={len(missing)} unexpected={len(unexpected)}")

# 6) zero-shot classifier (mri)
from hlip.zeroshot_metadata_pubbrain5 import PROMPTS, TEMPLATES
classifier = build_zero_shot_classifier(
    model,
    tokenizer=tokenizer,
    classnames=PROMPTS['prompt'],
    templates=TEMPLATES['template'],
    num_classes_per_batch=None,
    device=DEVICE,
    use_tqdm=False,
)

# 7) data and inference
study_dir = repo_dir / 'docs' / 'BraTS-Glioma'  # adjust if your example lives elsewhere
image = loader(str(study_dir), num_slices=48).to(DEVICE, non_blocking=True)

model.eval()
with torch.no_grad():
    output = model(image=image)
    image_features = output['image_features']
    # use the model's logit scale (exp)
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ classifier
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

for i, prompt in enumerate(PROMPTS['prompt']):
    print(f'{prompt}: {probs[0, i]:.4f}')