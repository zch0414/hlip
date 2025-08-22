import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import logging

from hlip.zeroshot_metadata_ct_rate import PROMPTS
from hlip_test.zeroshot_ct_rate import zero_shot as run_ct_rate


def zero_shot_eval(model, data, epoch, args, tokenizer):
    if 'zeroshot-ct-rate' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module
    if args.zeroshot_template != 'organ':
        PROMPTS["Lung nodule"] = ("Not lung nodule", "Lung nodule")
        PROMPTS["Lung opacity"] = ("Not lung opacity", "Lung opacity")

    
    logging.info('Starting Zero-Shot CT-RATE.')
    result = run_ct_rate(model, tokenizer, data['zeroshot-ct-rate'].dataloader, args)
    logging.info('Finished Zero-Shot CT-RATE.')
    return result['* mean']