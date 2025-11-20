import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast
from open_clip_train.distributed import is_master, init_distributed_device, all_gather_object

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from huggingface_hub import snapshot_download

from hlip import visual_encoder, visual_encoder_rope
from hlip.zeroshot_metadata_pubbrain5 import CLASSNAMES, TEMPLATES, PROMPTS


# arguments
def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    parser.add_argument('--model', default='clip_vit_base_multiscan_h2_token1176', type=str)
    parser.add_argument('--resume', default='/pretrained/brainmri_clip_vit_base_multiscan_h2_token1176.pt', type=str)
    parser.add_argument('--huggingface', default=None, type=str, help='HF model repo id, e.g., Zch0414/hlip-2025-10-08')

    parser.add_argument('--data-root', default='/path/to/pub_brain_5')
    parser.add_argument('--input-file', default='../../data/pub_brain_5.csv')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--save', default='', type=str)

    # hack argument
    parser.add_argument('--horovod', default=False, action='store_true')
    return parser


# random
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


# data
class PubBrain5Dataset(Dataset):
    def __init__(
        self,
        data_root,
        input_file,
    ):
        self.data_root = data_root
        self.studies = []
        df = pd.read_csv(input_file)
        for _, row in df.iterrows():
            if len(os.listdir(os.path.join(self.data_root, row['study']))):
                self.studies.append((row['study'] , row[CLASSNAMES].astype(int).tolist()))

        # debug
        # self.studies = self.studies[: 32]

        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study, target = self.studies[idx]

        image = []
        for scan in [os.path.join(self.data_root, study, p) for p in os.listdir(os.path.join(self.data_root, study))]:
            # load in img
            img = torch.load(scan, weights_only=True)                
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img.float() / 255.0
            _, h, w = img.shape

            # padding to the longest side.                
            size = max(h, w)
            pad_h = size - h; pad_w = size - w
            left = pad_w // 2; right = pad_w - left; top = pad_h // 2; bottom = pad_h - top
            img = torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=0)

            # resize to 256, crop to 224
            img = torch.nn.functional.interpolate(img[None, ...], size=(256, 256), mode='bilinear')[0]
            img = torch.nn.functional.interpolate(img[None, None, ...], size=(48, 256, 256), mode='nearest-exact')[0, 0]
            img = img[:, 16:240, 16:240]

            # normalize
            img = self.normalizer(img[None, ...])
            image.append(img) 
        
        # NOTE: convert image to torch.float16 by default
        return {'image': torch.stack(image, dim=0).to(dtype=torch.float16), 'target': torch.as_tensor(target, dtype=torch.long)}


def get_data(data_root, input_file, workers, distributed):
    dataset = PubBrain5Dataset(data_root, input_file)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=1, # only support 1 during evaluation; the speed bottleneck is data loading
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


# metric
def compute_pubbrain5_metrics(ground_truth, prediction):
    assert prediction.shape == ground_truth.shape and prediction.shape[1] == 5, "Expected [N, 5] inputs."
    results = {}

    # ---------------- anomaly ----------------
    aucs, f1ws, bas = [], [], []
    disease_cols = {'stroke': 1, 'glioma': 2, 'meningioma': 3, 'metastasis': 4}

    for disease, c in disease_cols.items():
        mask = (ground_truth[:, 0] + ground_truth[:, c] == 1.0)
        y_true = ground_truth[mask, c].to(torch.int64).numpy()
        logits = prediction[mask][:, [0, c]]
        logits = torch.softmax(logits, dim=1).numpy()
        y_score = logits[:, 1]
        y_pred = (y_score >= 0.5).astype(int)

        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ba = balanced_accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan

        results[f'auc (pubbrain5_anomaly_{disease})'] = float(auc)
        results[f'weighted_f1 (pubbrain5_anomaly_{disease})'] = float(f1w)
        results[f'balanced_acc (pubbrain5_anomaly_{disease})'] = float(ba)

        aucs.append(auc); f1ws.append(f1w); bas.append(ba)

    results.update({
        'auc (pubbrain5_anomaly)'        : float(np.nanmean(aucs)),
        'weighted_f1 (pubbrain5_anomaly)': float(np.nanmean(f1ws)),
        'balanced_acc (pubbrain5_anomaly)': float(np.nanmean(bas)),
    })

    # ---------------- tumor (glioma/meningioma/metastasis) ----------------
    tumor_cols = [2, 3, 4]
    tumor_names = ['glioma', 'meningioma', 'metastasis']

    mask = (ground_truth[:, tumor_cols].sum(dim=1) == 1.0)
    y_true_t = torch.argmax(ground_truth[mask][:, tumor_cols], dim=1).numpy()  # 0..2
    y_score_t = torch.softmax(prediction[mask][:, tumor_cols], dim=1).numpy()
    y_pred_t = y_score_t.argmax(axis=1)

    # overall tumor metrics (3-way)
    f1w = f1_score(y_true_t, y_pred_t, average="weighted", zero_division=0)
    ba  = balanced_accuracy_score(y_true_t, y_pred_t)
    try:
        y_true_oh = np.eye(3, dtype=int)[y_true_t]
        auc = roc_auc_score(y_true_oh, y_score_t, average="macro", multi_class="ovr")
    except ValueError:
        auc = np.nan

    results.update({
        'auc (pubbrain5_tumor)'        : float(auc),
        'weighted_f1 (pubbrain5_tumor)': float(f1w),
        'balanced_acc (pubbrain5_tumor)': float(ba),
    })

    # per-tumor subtype (one-vs-rest)
    for i, name in enumerate(tumor_names):
        y_true_bin = (y_true_t == i).astype(int)
        y_score_bin = y_score_t[:, i]
        y_pred_bin = (y_score_bin >= 0.5).astype(int)

        f1w_i = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
        ba_i  = balanced_accuracy_score(y_true_bin, y_pred_bin)
        try:
            auc_i = roc_auc_score(y_true_bin, y_score_bin)
        except ValueError:
            auc_i = np.nan

        results[f'auc (pubbrain5_tumor_{name})']        = float(auc_i)
        results[f'weighted_f1 (pubbrain5_tumor_{name})'] = float(f1w_i)
        results[f'balanced_acc (pubbrain5_tumor_{name})'] = float(ba_i)

    # ---------------- disease (normal/stroke/glioma/meningioma/metastasis) ----------------
    disease_names = ['normal', 'stroke', 'glioma', 'meningioma', 'metastasis']

    y_true_d = torch.argmax(ground_truth, dim=1).numpy()        # 0..4
    y_score_d = torch.softmax(prediction, dim=1).numpy()
    y_pred_d = y_score_d.argmax(axis=1)

    # overall disease metrics (5-way)
    f1w = f1_score(y_true_d, y_pred_d, average="weighted", zero_division=0)
    ba  = balanced_accuracy_score(y_true_d, y_pred_d)
    try:
        y_true_oh = ground_truth.numpy()  # already one-hot
        auc = roc_auc_score(y_true_oh, y_score_d, average="macro", multi_class="ovr")
    except ValueError:
        auc = np.nan

    results.update({
        'auc (pubbrain5_disease)'        : float(auc),
        'weighted_f1 (pubbrain5_disease)': float(f1w),
        'balanced_acc (pubbrain5_disease)': float(ba),
    })

    # per-disease class (one-vs-rest)
    for i, name in enumerate(disease_names):
        y_true_bin = (y_true_d == i).astype(int)
        y_score_bin = y_score_d[:, i]
        y_pred_bin = (y_score_bin >= 0.5).astype(int)

        f1w_i = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
        ba_i  = balanced_accuracy_score(y_true_bin, y_pred_bin)
        try:
            auc_i = roc_auc_score(y_true_bin, y_score_bin)
        except ValueError:
            auc_i = np.nan

        results[f'auc (pubbrain5_disease_{name})']         = float(auc_i)
        results[f'weighted_f1 (pubbrain5_disease_{name})'] = float(f1w_i)
        results[f'balanced_acc (pubbrain5_disease_{name})'] = float(ba_i)

    return results


# run
def run(model, tokenizer, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')

    # build classifier
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=PROMPTS['prompt'],
            templates=TEMPLATES['template'],
            num_classes_per_batch=None, # all
            device=device,
            use_tqdm=False,
        )

    prediction = []
    ground_truth = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader), disable=not is_master(args)):
            image = batch['image'].to(device=device, dtype=input_dtype, non_blocking=True)
            ground_truth.append(batch['target'].cpu())

            with autocast():
                model_out = model(image=image)
                logit_scale = model_out['logit_scale']
                image_features = model_out['image_features']
                logits_per_image = logit_scale * image_features @ classifier
                prediction.append(logits_per_image.detach().cpu())
            
    return torch.cat(ground_truth, dim=0), torch.cat(prediction, dim=0)


# main
def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)
    if args.distributed:
        print(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        print(f'Running with a single process. Device {args.device}.')
    random_seed(0, 0)

    # create model
    for _c in os.listdir('../hlip/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../hlip/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)
    
    # load checkpoint
    if args.huggingface is not None:
        local_dir = snapshot_download(repo_id=args.huggingface)
        bin_path = Path(local_dir) / "pytorch_model.bin"
        if not bin_path.is_file():
            raise FileNotFoundError(f"Expected pytorch_model.bin in HF repo {args.huggingface}, but not found at {bin_path}")
        state_dict = torch.load(bin_path, map_location="cpu")
        sd = state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict
    else:
        checkpoint = pt_load(args.resume, map_location='cpu')
        sd = checkpoint['state_dict']
        # sd = {k.replace('module.visual.trunk.series_posemb', 'module.visual.trunk.sequential_posemb'): v for k, v in sd.items()}
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        
    model.load_state_dict(sd)
    tokenizer = get_tokenizer(args.model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], static_graph=False)

    # create data
    dataloader = get_data(
        data_root=args.data_root, 
        input_file=args.input_file,
        workers=args.workers, 
        distributed=args.distributed
    )

    # run
    if args.distributed and not args.horovod:
        model = model.module
        
    model.eval()
    ground_truth, prediction = run(model, tokenizer, dataloader, args)
    if args.distributed:
        prediction = all_gather_object(args, prediction)
        ground_truth = all_gather_object(args, ground_truth)
    else:
        prediction = [prediction]
        ground_truth = [ground_truth]

    if is_master(args):
        prediction = torch.cat(prediction, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        print(f'Compute metrics on {prediction.shape[0]} cases.')
        results = compute_pubbrain5_metrics(ground_truth, prediction)
        for k, v in results.items():
            print(f'{k}: {v}')
        if args.save:
            p = Path(args.save)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:   # append one JSON per line
                f.write(json.dumps(results, ensure_ascii=False))
                f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)