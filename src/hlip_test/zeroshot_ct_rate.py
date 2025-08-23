import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import math
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from hlip import visual_encoder
from hlip.zeroshot_metadata_ct_rate import CLASSNAMES, ORGANS, TEMPLATES, PROMPTS



def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    parser.add_argument('--model', default='clip_vit_base_singlescan_h2_token2744', type=str)
    parser.add_argument('--use-cxr-bert', default=False, action='store_true')
    parser.add_argument('--lora-text', default=False, action='store_true')
    parser.add_argument('--resume', default='/pretrained/chestct_clip_vit_base_singlescan_h2_token2744.pt', type=str)

    parser.add_argument('--data-root', default='/data/ct_rate/')
    parser.add_argument('--zeroshot-ct-rate', default='../../data/ct_rate/metafiles/valid_labels.csv', type=str)
    parser.add_argument('--input-info', nargs='+', default=["-1150", "350", "crop"])
    parser.add_argument('--zeroshot-template', default='volume', type=str)

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--workers', default=4, type=int)

    return parser


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_data(args, preprocess_fn=None):
    class ZeroShotDataset(Dataset):
        def __init__(
            self,
            root, input_filename, input_info,
            transform=None,
        ):
            self.cts = []
            df = pd.read_csv(input_filename)
            for _, row in df.iterrows():
                recon = row['VolumeName']
                recon = recon.rsplit('.', 2)[0]
                self.cts.append((os.path.join(root, 'valid', recon.rsplit('_', 2)[0], recon.rsplit('_', 1)[0], recon + '.pt'), row[CLASSNAMES].astype(int).tolist()))
            
            self.input_info = (float(input_info[0]), float(input_info[1]), str(input_info[2]))
            self.transform = transform

        def __len__(self):
            return len(self.cts)

        def __getitem__(self, idx):
            recon, target = self.cts[idx]

            img = torch.load(recon, weights_only=True)
            img = (img - self.input_info[0]) / (self.input_info[1] - self.input_info[0])
            img = torch.clip(img, 0., 1.)
            img = img[None, ...].float()

            if self.transform:
                img = self.transform(img)
                img = torch.as_tensor(img).float()
            else: 
                if self.input_info[2] == "crop":
                    _, d, h, w = img.shape
                    pad_d = max(112 - d, 0)
                    pad_h = max(336 - h, 0)
                    pad_w = max(336 - w, 0)
                    pad_d1, pad_d2 = pad_d // 2, pad_d - pad_d // 2
                    pad_h1, pad_h2 = pad_h // 2, pad_h - pad_h // 2
                    pad_w1, pad_w2 = pad_w // 2, pad_w - pad_w // 2
                    img = torch.nn.functional.pad(
                        img[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2, pad_d1, pad_d2),
                        mode='constant', 
                        value=0
                    ).squeeze(0)
                    
                    _, d, h, w = img.shape
                    start_d = (d - 112) // 2
                    start_h = (h - 336) // 2
                    start_w = (w - 336) // 2
                    img = img[
                        :, 
                        start_d:start_d + 112,
                        start_h:start_h + 336,
                        start_w:start_w + 336
                    ]
                
                elif self.input_info[2] == "resize":
                    img = torch.nn.functional.interpolate(img[None, ...], size=(112, 336, 336), mode='trilinear').squeeze(0)
                
                else:
                    raise NotImplementedError

            # normalize
            normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
            img = normalizer(img)

            return recon, img[None, ...], torch.as_tensor(target)
    

    dataset = ZeroShotDataset(
        args.data_root, args.zeroshot_ct_rate, args.input_info,
        preprocess_fn
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    return dataloader


def find_threshold(y_true, y_score):
    """
    Copy from https://github.com/alibaba-damo-academy/fvlm/blob/d768ec1546fb825fcc9ea9b3e7b2754a69f870c1/calc_metrics.py#L8C1-L8C32
    Finds the optimal threshold for binary classification based on ROC curve.

    Args:
        y_true (numpy.ndarray): True labels.
        y_score (numpy.ndarray): Predicted probabilities.

    Returns:
        float: Optimal threshold.
    """

    best_threshold = 0
    best_roc = 10000

    # Iterate over potential thresholds
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(int)
        confusion = confusion_matrix(y_true, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        TP_r = TP / (TP + FN)
        FP_r = FP / (FP + TN)
        curr_roc = math.sqrt(((1 - TP_r) ** 2) + (FP_r ** 2))
        if curr_roc <= best_roc:
            best_roc = curr_roc
            best_threshold = threshold

    return best_threshold


def zero_shot(model, tokenizer, dataloader, args):
    model.eval()
    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')

    # build classifier
    with autocast():
        classifier = {}
        for key in CLASSNAMES:
            classifier.update(
                {
                    key: build_zero_shot_classifier(
                            model,
                            tokenizer=tokenizer,
                            classnames=PROMPTS[key],
                            templates=TEMPLATES[ORGANS[key]] if args.zeroshot_template == 'organ' else TEMPLATES[args.zeroshot_template],
                            num_classes_per_batch=None, # all
                            device=device,
                            use_tqdm=False,
                        )
                }
            )

    with torch.inference_mode():
        columns = ['recon'] + CLASSNAMES
        rows = []
        labels = {key: [] for key in CLASSNAMES}
        logits = {key: [] for key in CLASSNAMES}
        preds = {key: [] for key in CLASSNAMES}
        
        for batch in tqdm(dataloader, total=len(dataloader)):
            recon, image, target = batch
            image = image.to(device=device, dtype=input_dtype)
            target = target.to(device)
            row = []
            row.append(recon[0])
            
            for idx in range(target.shape[1]):
                labels[CLASSNAMES[idx]].append(target[0, idx].cpu().float().item())
            
            with autocast():
                output = model(image=image)
                image_features = output['image_features']
                logit_scale = output['logit_scale']
                
                # predict
                for key, value in classifier.items():
                    logits_per_image = logit_scale * image_features @ value
                    logits_per_image = logits_per_image.softmax(dim=1)
                    row.append(logits_per_image[0, 1].cpu().float().item())
                    logits[key].append(logits_per_image[0, 1].cpu().float().item())
                    preds[key].append(logits_per_image.argmax(-1).cpu().float().item())

            rows.append(row)

        results_dir = f"./results/ct_rate/{args.model}/"
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(os.path.join(results_dir, f'logits_{args.zeroshot_template}.csv'), index=False)
        print(f"CSV file created: {os.path.join(results_dir, f'logits_{args.zeroshot_template}.csv')}")

        results = {key: {} for key in CLASSNAMES}
        mean_balanced_acc, mean_weighted_f1, mean_recall, mean_precision = 0., 0., 0., 0.
        for key in CLASSNAMES:
            balanced_acc = balanced_accuracy_score(np.array(labels[key]), np.array(preds[key]))
            mean_balanced_acc += balanced_acc / len(CLASSNAMES)

            weighted_f1 = f1_score(np.array(labels[key]), np.array(preds[key]), average='weighted') 
            mean_weighted_f1 += weighted_f1 / len(CLASSNAMES)

            recall = recall_score(np.array(labels[key]), np.array(preds[key])) 
            mean_recall += recall / len(CLASSNAMES)

            precision = precision_score(np.array(labels[key]), np.array(preds[key])) 
            mean_precision += precision / len(CLASSNAMES)

            results[key].update({
                'acc (balanced)': balanced_acc,
                'f1 (weighted)': weighted_f1,
                'recall': recall,
                'precision': precision,
            })
        results['mean'] = {
            'mean acc (balanced)': mean_balanced_acc,
            'mean f1 (weighted)': mean_weighted_f1,
            'mean recall': mean_recall,
            'mean precision': mean_precision,
        }

        mean_auc, mean_acc, mean_weighted_f1, mean_recall, mean_precision = 0., 0., 0., 0., 0.
        for key in CLASSNAMES:
            threshold = find_threshold(np.array(labels[key]), np.array(logits[key]))

            auc = roc_auc_score(np.array(labels[key]), np.array(logits[key])) 
            mean_auc += auc / len(CLASSNAMES)

            acc = accuracy_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int)) 
            mean_acc += acc / len(CLASSNAMES)

            weighted_f1 = f1_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int), average='weighted')
            mean_weighted_f1 += weighted_f1 / len(CLASSNAMES)

            recall = recall_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int)) 
            mean_recall += recall / len(CLASSNAMES)

            precision = precision_score(np.array(labels[key]), (np.array(logits[key]) > threshold).astype(int)) 
            mean_precision += precision / len(CLASSNAMES)

            results[key].update({
                'auc': auc,
                '* acc (balanced)': acc,
                '* f1 (weighted)': weighted_f1,
                '* recall': recall,
                '* precision': precision,
            })
        results['* mean'] = {
            'mean auc': mean_auc,
            '* mean acc (balanced)': mean_acc,
            '* mean f1 (weighted)': mean_weighted_f1,
            '* mean recall': mean_recall,
            '* mean precision': mean_precision,
        }

    return results


def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if args.zeroshot_template != 'organ':
        PROMPTS["Lung nodule"] = ("Not lung nodule", "Lung nodule")
        PROMPTS["Lung opacity"] = ("Not lung opacity", "Lung opacity")

    random_seed(0, 0)

    # create model
    for _c in os.listdir('../hlip/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../hlip/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)
    
    # replace with cxr_bert
    if args.use_cxr_bert:
        from transformers import AutoModel
        cxr_bert = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True).bert
        if args.lora_text:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=["query", "value"],
                lora_dropout=0.0,
                bias="none",
            )
            cxr_bert = get_peft_model(cxr_bert, lora_config)
            for n, p in cxr_bert.named_parameters():
                p.requires_grad = (not args.lock_text_freeze_layer_norm) if "LayerNorm" in n.split(".") else False
        cxr_bert.to(device=args.device)
        model.text.transformer = cxr_bert

    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)

    # create dataset
    data = get_data(args, None)

    # zero shot
    results = zero_shot(model, tokenizer, data, args)
    results_dir = f"./results/ct_rate/{args.model}/"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'results_{args.zeroshot_template}.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)