import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import logging
import torch
from open_clip_train.distributed import is_master, all_gather_object

from hlip_test.zeroshot_mri import run as run_mri
from hlip_test.zeroshot_mri import compute_metrics as compute_mri_metrics
from hlip_test.zeroshot_ct import run as run_ct
from hlip_test.zeroshot_ct import compute_metrics as compute_ct_metrics


def zero_shot_eval(model, data, epoch, args, tokenizer):
    if 'mri' not in data and 'ct' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    if 'mri' in data:
        ground_truth, prediction = run_mri(model, tokenizer, data['mri'], args)
        mri_ground_truth = all_gather_object(args, ground_truth)
        mri_prediction = all_gather_object(args, prediction)
    if 'ct' in data:
        ground_truth, prediction = run_ct(model, tokenizer, data['ct'], args)
        ct_ground_truth = all_gather_object(args, ground_truth)
        ct_prediction = all_gather_object(args, prediction)

    if not is_master(args):
        return {}
    
    results = {}
    if 'mri' in data:
        prediction = torch.cat(mri_prediction, dim=0)
        ground_truth = torch.cat(mri_ground_truth, dim=0)
        results.update(compute_mri_metrics(ground_truth, prediction))
    if 'ct' in data:
        prediction = torch.cat(ct_prediction, dim=0)
        ground_truth = torch.cat(ct_ground_truth, dim=0)
        results.update(compute_ct_metrics(ground_truth, prediction))

    return results