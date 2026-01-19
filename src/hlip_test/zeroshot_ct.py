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
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast
from open_clip_train.distributed import is_master, init_distributed_device, all_gather_object

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from hlip import visual_encoder, visual_encoder_rope
from hlip.zeroshot_metadata_ct import CLASSNAMES, TEMPLATES, PROMPTS


# arguments
def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    parser.add_argument('--model', default='clip_vit_base_multiscan_h2_token1176', type=str)
    parser.add_argument('--resume', default='/pretrained/headct_clip_vit_base_multiscan_h2_token1176.pt', type=str)
    
    parser.add_argument('--data-root', default='/path/to/ct')
    parser.add_argument('--input-file', default='../../data/ct.csv')
    parser.add_argument('--simple-template', default=False, action='store_true')
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
class StudyDataset(Dataset):
    def __init__(
        self,
        data_root,
        input_file,
    ):
        self.data_root = data_root
        self.studies = []
        df = pd.read_csv(input_file)
        for _, row in df.iterrows():
            self.studies.append((row['study'], row[CLASSNAMES].astype(int).tolist()))
        
        # debug
        # self.studies = self.studies[: 32]
        
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study, target = self.studies[idx]

        image = []
        for scan in [os.path.join(self.data_root, study, p, 'img.pt') for p in os.listdir(os.path.join(self.data_root, study))]:
            img = torch.load(scan, weights_only=True)
            img = img.float() / 255.0
            img = self.normalizer(img[None, ...])
            image.append(img) 
        
        # NOTE: convert image to torch.float16 by default
        return {'image': torch.stack(image, dim=0).to(dtype=torch.float16), 'target': torch.as_tensor(target, dtype=torch.long)}


def get_data(data_root, input_file, workers, distributed):
    dataset = StudyDataset(data_root, input_file)
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
def compute_metrics(ground_truth, prediction):
    assert prediction.shape == ground_truth.shape
    results = {}
    
    try:
        auc = roc_auc_score(ground_truth, prediction, average="macro", multi_class="ovr")
    except ValueError:
        auc = np.nan

    results.update({
        'auc (ct)': float(auc),
    })
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
            templates=(lambda c: f'This study shows: {c}.',) if getattr(args, "simple_template", False) else TEMPLATES['template'],
            num_classes_per_batch=None, # all
            device=device,
            use_tqdm=False,
        )

    prediction = []
    ground_truth = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader), disable=not is_master(args)):
            image = batch['image'].to(device=device, dtype=input_dtype, non_blocking=True)
            ground_truth.append(batch['target'].to(device=device, non_blocking=True))

            with autocast():
                model_out = model(image=image)
                image_features = model_out['image_features'][:, 0, :]
                logits_per_image = image_features @ classifier
                prediction.append(logits_per_image)

    return torch.cat(ground_truth, dim=0).cpu(), torch.cat(prediction, dim=0).cpu()


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
    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']

    # For internal code base
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
    prediction = all_gather_object(args, prediction)
    ground_truth = all_gather_object(args, ground_truth)

    if is_master(args):
        prediction = torch.cat(prediction, dim=0)
        ground_truth = torch.cat(ground_truth, dim=0)

        print(f'Compute metrics on {prediction.shape[0]} cases.')
        results = compute_metrics(ground_truth, prediction)
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