import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, balanced_accuracy_score

from open_clip import create_model_and_transforms, get_tokenizer, get_input_dtype, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load
from open_clip_train.precision import get_autocast

import torch
from torch.utils.data import Dataset, DataLoader

from hlip import visual_encoder
from hlip.zeroshot_metadata_pub_brain_5 import CLASSNAMES, TEMPLATES, PROMPTS


def get_args_parser():
    parser = argparse.ArgumentParser('Perform Zero-shot', add_help=False)
    parser.add_argument('--model', default='clip_vit_base_multiscan_h2_token1176', type=str)
    parser.add_argument('--resume', default='/pretrained/brainmri_clip_vit_base_multiscan_h2_token1176.pt', type=str)
    
    parser.add_argument('--tasks', nargs='+', default=['stroke', 'glioma', 'meningioma', 'metastasis', 'tumor', 'disease'])
    parser.add_argument('--embed-root', default='/path/to/pub_brain_5_embed')
    parser.add_argument('--input-filename', default='../../data/pub_brain_5/pub_brain_5.csv')
    parser.add_argument('--num-slices', default=144, type=int)
    parser.add_argument('--zeroshot-prompt', default='prompt', type=str)
    parser.add_argument('--zeroshot-template', default='template', type=str)
    
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--workers', default=16, type=int)
    return parser


def get_data(args):
    class ZeroShotDataset(Dataset):
        def __init__(
            self,
            embed_root,
            input_filename,
        ):
            self.embed_root = embed_root
            self.studies = []
            df = pd.read_csv(input_filename)
            for _, row in df.iterrows():
                if len(os.listdir(os.path.join(self.embed_root, args.model, str(args.num_slices), row['study']))):
                    self.studies.append((row['study'] , row[CLASSNAMES].astype(int).tolist()))

        def __len__(self):
            return len(self.studies)

        def __getitem__(self, idx):
            study, target = self.studies[idx]

            # load in image_features and logit_scale
            embed_dir = os.path.join(self.embed_root, args.model, str(args.num_slices), study)
            image_features = torch.load(os.path.join(embed_dir, 'image_features.pt'), weights_only=True)
            logit_scale = torch.load(os.path.join(embed_dir, 'logit_scale.pt'), weights_only=True)
            return study, image_features, logit_scale, torch.as_tensor(target)


    dataset = ZeroShotDataset(args.embed_root, args.input_filename)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloader


def zero_shot(model, tokenizer, dataloader, args):
    model.eval()
    device = torch.device(args.device)
    autocast = get_autocast('amp', device_type=device.type)
    input_dtype = get_input_dtype('amp')

    # build classifier
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=PROMPTS[args.zeroshot_prompt],
            templates=TEMPLATES[args.zeroshot_template],
            num_classes_per_batch=None, # all
            device=device,
            use_tqdm=False,
        )

    # stroke detection (2-way)
    if 'stroke' in args.tasks:
        with torch.inference_mode():
            # save the logits
            columns = ['study'] + ['stroke']
            rows = []
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for batch in tqdm(dataloader, total=len(dataloader)):
                study, image_features, logit_scale, target = batch
                if 'stroke' not in study[0] and 'open_bhb' not in study[0]:
                    continue
                row = []
                row.append(study[0])
                ground_truth.append(target.numpy()[0, 1])

                image_features = image_features.to(device=device, dtype=input_dtype)
                logit_scale = logit_scale.to(device=device, dtype=input_dtype)
                
                with autocast():
                    # predict
                    logits_per_image = logit_scale[0] * image_features[0] @ classifier
                    logits_per_image = logits_per_image[:, [0, 1]].softmax(dim=-1) # [0, 1] here refers to the health and stroke.
                    logits_per_image = logits_per_image.cpu().float().numpy()
                    row.append(logits_per_image[0, 1])
                    logits.append(logits_per_image[0, 1])
                    preds.append(logits_per_image.argmax(-1))
                rows.append(row)

            results_dir = f"./results/{args.input_filename.split('/')[-1].split('.')[0]}/{args.model}/{args.num_slices}/"
            os.makedirs(results_dir, exist_ok=True)
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(os.path.join(results_dir, f'stroke_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv'), index=False)
            print(f"CSV file created: {os.path.join(results_dir, f'stroke_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv')}")

            # metrics
            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Stroke Detection (2-way): Balanced Acc: {acc}; Weghted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # Glioma detection (2-way)
    if 'glioma' in args.tasks:
        with torch.inference_mode():
            # save the logits
            columns = ['study'] + ['glioma']
            rows = []
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for batch in tqdm(dataloader, total=len(dataloader)):
                study, image_features, logit_scale, target = batch
                if 'glioma' not in study[0] and 'open_bhb' not in study[0]:
                    continue
                row = []
                row.append(study[0])
                ground_truth.append(target.numpy()[0, 2])

                image_features = image_features.to(device=device, dtype=input_dtype)
                logit_scale = logit_scale.to(device=device, dtype=input_dtype)
                
                with autocast():
                    # predict
                    logits_per_image = logit_scale[0] * image_features[0] @ classifier
                    logits_per_image = logits_per_image[:, [0, 2]].softmax(dim=-1) # [0, 2] here refers the health and glioma.
                    logits_per_image = logits_per_image.cpu().float().numpy()
                    row.append(logits_per_image[0, 1])
                    logits.append(logits_per_image[0, 1])
                    preds.append(logits_per_image.argmax(-1))
                rows.append(row)

            results_dir = f"./results/{args.input_filename.split('/')[-1].split('.')[0]}/{args.model}/{args.num_slices}/"
            os.makedirs(results_dir, exist_ok=True)
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(os.path.join(results_dir, f'glioma_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv'), index=False)
            print(f"CSV file created: {os.path.join(results_dir, f'glioma_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv')}")

            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Glioma Detection (2-way): Balanced Acc: {acc}; Weighted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # Meningioma detection (2-way)
    if 'meningioma' in args.tasks:
        with torch.inference_mode():
            # save the logits
            columns = ['study'] + ['meningioma']
            rows = []
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for batch in tqdm(dataloader, total=len(dataloader)):
                study, image_features, logit_scale, target = batch
                if 'meningioma' not in study[0] and 'open_bhb' not in study[0]:
                    continue
                row = []
                row.append(study[0])
                ground_truth.append(target.numpy()[0, 3])

                image_features = image_features.to(device=device, dtype=input_dtype)
                logit_scale = logit_scale.to(device=device, dtype=input_dtype)
                
                with autocast():
                    # predict
                    logits_per_image = logit_scale[0] * image_features[0] @ classifier
                    logits_per_image = logits_per_image[:, [0, 3]].softmax(dim=-1) # [0, 3] here refers to the health and meningioma
                    logits_per_image = logits_per_image.cpu().float().numpy()
                    row.append(logits_per_image[0, 1])
                    logits.append(logits_per_image[0, 1])
                    preds.append(logits_per_image.argmax(-1))
                rows.append(row)

            results_dir = f"./results/{args.input_filename.split('/')[-1].split('.')[0]}/{args.model}/{args.num_slices}/"
            os.makedirs(results_dir, exist_ok=True)
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(os.path.join(results_dir, f'meningioma_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv'), index=False)
            print(f"CSV file created: {os.path.join(results_dir, f'meningioma_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv')}")

            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Meningioma Detection (2-way): Balanced Acc: {acc}; Weighted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # Metastasis detection (2-way)
    if 'metastasis' in args.tasks:
        with torch.inference_mode():
            # save the logits
            columns = ['study'] + ['metastasis']
            rows = []
            ground_truth = []
            logits = []
            preds = []

            # start testing
            for batch in tqdm(dataloader, total=len(dataloader)):
                study, image_features, logit_scale, target = batch
                if 'metastasis' not in study[0] and 'nyu_mets' not in study[0] and 'ucsf_mets' not in study[0] and 'open_bhb' not in study[0]:
                    continue
                row = []
                row.append(study[0])
                ground_truth.append(target.numpy()[0, 4])

                image_features = image_features.to(device=device, dtype=input_dtype)
                logit_scale = logit_scale.to(device=device, dtype=input_dtype)
                
                with autocast():
                    # predict
                    logits_per_image = logit_scale[0] * image_features[0] @ classifier
                    logits_per_image = logits_per_image[:, [0, 4]].softmax(dim=-1) # [0, 4] here refers to the health and metastasis.
                    logits_per_image = logits_per_image.cpu().float().numpy()
                    row.append(logits_per_image[0, 1])
                    logits.append(logits_per_image[0, 1])
                    preds.append(logits_per_image.argmax(-1))
                rows.append(row)

            results_dir = f"./results/{args.input_filename.split('/')[-1].split('.')[0]}/{args.model}/{args.num_slices}/"
            os.makedirs(results_dir, exist_ok=True)
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(os.path.join(results_dir, f'metastasis_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv'), index=False)
            print(f"CSV file created: {os.path.join(results_dir, f'metastasis_detection_{args.zeroshot_template}_{args.zeroshot_prompt}.csv')}")

            ground_truth = np.array(ground_truth)
            logits = np.array(logits)
            preds = np.array(preds)

            acc = balanced_accuracy_score(ground_truth, preds)
            f1 = f1_score(ground_truth, preds, average='weighted')
            recall = recall_score(ground_truth, preds)
            precision = precision_score(ground_truth, preds)
            auc = roc_auc_score(ground_truth, logits)
            print(f"Metastasis Detection (2-way): Balanced Acc: {acc}; Weighted F1: {f1}; Recall: {recall}; Precision: {precision}; Macro AUC: {auc}")

    # brain tumor classification (3-way)
    if 'tumor' in args.tasks:
        with torch.inference_mode():
            # save the logits
            columns = ['study'] + CLASSNAMES[2:]
            rows = []
            ground_truth = []
            logits = []

            # start testing
            for batch in tqdm(dataloader, total=len(dataloader)):
                study, image_features, logit_scale, target = batch
                if 'brats23' not in study[0] and 'nyu_mets' not in study[0] and 'ucsf_mets' not in study[0]:
                    continue
                row = []
                row.append(study[0])
                ground_truth.append(target.numpy()[:, 2:]) #  [1, num_classes]

                image_features = image_features.to(device=device, dtype=input_dtype)
                logit_scale = logit_scale.to(device=device, dtype=input_dtype)
                
                with autocast():
                    # predict
                    logits_per_image = logit_scale[0] * image_features[0] @ classifier
                    logits_per_image = logits_per_image[:, [2, 3, 4]].softmax(dim=-1)
                    logits_per_image = logits_per_image.cpu().float().numpy()
                    row.extend(logits_per_image[0].tolist())
                    logits.append(logits_per_image)
                rows.append(row)

            results_dir = f"./results/{args.input_filename.split('/')[-1].split('.')[0]}/{args.model}/{args.num_slices}/"
            os.makedirs(results_dir, exist_ok=True)
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(os.path.join(results_dir, f'tumor_classification_{args.zeroshot_template}_{args.zeroshot_prompt}.csv'), index=False)
            print(f"CSV file created: {os.path.join(results_dir, f'tumor_classification_{args.zeroshot_template}_{args.zeroshot_prompt}.csv')}")

            ground_truth = np.concatenate(ground_truth, axis=0)
            logits = np.concatenate(logits, axis=0)
            acc = balanced_accuracy_score(ground_truth.argmax(-1), logits.argmax(-1))
            auc = roc_auc_score(ground_truth, logits)
            print(f"Tumor Classification (3-way): Balanced Acc: {acc}; Macro AUC: {auc}")

    # brain disease classification (5-way)
    if 'disease' in args.tasks:
        with torch.inference_mode():
            # save the logits
            columns = ['study'] + CLASSNAMES
            rows = []
            ground_truth = []
            logits = []

            # start testing
            for batch in tqdm(dataloader, total=len(dataloader)):
                study, image_features, logit_scale, target = batch
                row = []
                row.append(study[0])
                ground_truth.append(target.numpy()) #  [1, num_classes]

                image_features = image_features.to(device=device, dtype=input_dtype)
                logit_scale = logit_scale.to(device=device, dtype=input_dtype)
                
                with autocast():
                    # predict
                    logits_per_image = logit_scale[0] * image_features[0] @ classifier
                    logits_per_image = logits_per_image.softmax(dim=-1)
                    logits_per_image = logits_per_image.cpu().float().numpy()
                    row.extend(logits_per_image[0].tolist())
                    logits.append(logits_per_image)
                rows.append(row)

            results_dir = f"./results/{args.input_filename.split('/')[-1].split('.')[0]}/{args.model}/{args.num_slices}/"
            os.makedirs(results_dir, exist_ok=True)
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(os.path.join(results_dir, f'disease_classification_{args.zeroshot_template}_{args.zeroshot_prompt}.csv'), index=False)
            print(f"CSV file created: {os.path.join(results_dir, f'disease_classification_{args.zeroshot_template}_{args.zeroshot_prompt}.csv')}")

            ground_truth = np.concatenate(ground_truth, axis=0)
            logits = np.concatenate(logits, axis=0)
            acc = balanced_accuracy_score(ground_truth.argmax(-1), logits.argmax(-1))
            auc = roc_auc_score(ground_truth, logits)
            print(f"Disease Classification (5-way): Balanced Acc: {acc}; Macro AUC: {auc}")
    return


def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Create model
    for _c in os.listdir('../hlip/model_configs/'):
        _m, _e = os.path.splitext(_c)
        if _e.lower() == '.json':
            with open(os.path.join('../hlip/model_configs/', _c), 'r') as f:
                model_cfg = json.load(f)
            _MODEL_CONFIGS[_m] = model_cfg
    model, _, _ = create_model_and_transforms(args.model, device=args.device, precision='amp', output_dict=True)
    checkpoint = pt_load(args.resume, map_location='cpu')
    sd = checkpoint['state_dict']
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    tokenizer = get_tokenizer(args.model)

    # zero shot
    dataloader = get_data(args)
    zero_shot(model, tokenizer, dataloader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)