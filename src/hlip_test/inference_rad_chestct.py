import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import json
import argparse

from open_clip import create_model_and_transforms, get_tokenizer, build_zero_shot_classifier
from open_clip.factory import _MODEL_CONFIGS
from open_clip_train.file_utils import pt_load

import torch
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from hlip import visual_encoder
from hlip.zeroshot_metadata_rad_chestct import ORGANS, PROMPTS, TEMPLATES


def get_args_parser():
    parser = argparse.ArgumentParser('Inference', add_help=False)
    
    # model
    parser.add_argument('--model', default='clip_vit_base_singlescan_h2_token2744', type=str)
    parser.add_argument('--lora-text', default=False, action='store_true')
    parser.add_argument('--use-cxr-bert', default=False, action='store_true')
    parser.add_argument('--resume', default='/pretrained/chestct_clip_vit_base_singlescan_h2_token2744.pt', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    
    # data
    parser.add_argument('--data', default='../../docs/tst32751/tst32751.pt')
    parser.add_argument('--zeroshot-prompt', default='volume', type=str)
    parser.add_argument('--input-info', nargs='+', default=["-1150", "350", "crop"])
    parser.add_argument('--target', default='Pulmonary fibrotic sequela', type=str)

    return parser


def loader(recon_path, args):
    input_info = (float(args.input_info[0]), float(args.input_info[1]), str(args.input_info[2]))
    img = torch.load(recon_path, weights_only=True)
    img = img[None, ...].float()
    img = (img - input_info[0]) / (input_info[1] - input_info[0])
    img = torch.clip(img, 0., 1.)

    if input_info[2] == "crop":
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

    elif input_info[2] == "resize":
        img = torch.nn.functional.interpolate(img[None, ...], size=(112, 336, 336), mode='trilinear').squeeze(0)

    else:
        raise NotImplementedError
    
    normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())
    img = normalizer(img)
    return img[None, None, ...] # [1, n=1, 1, d, h, w]

def inference(model, tokenizer, image, args):
    model.eval()
    device = torch.device(args.device)
    image = image.float().to(device=device)

    # build text embeddings
    classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=PROMPTS[args.target],
            templates=TEMPLATES['volume'] if args.zeroshot_prompt == 'volume' else TEMPLATES[ORGANS[args.target]],
            num_classes_per_batch=None, # all
            device=device,
            use_tqdm=False,
        )
    
    # inference
    output = model(image=image)
    logit_scale = output['logit_scale']
    image_features = output['image_features']
    logits_per_image = logit_scale * image_features @ classifier
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    for i, prompt in enumerate(PROMPTS[args.target]):
        print(f'{prompt}: {probs[0, i]:.4f}')
    return


def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    if args.zeroshot_prompt == 'volume':
        PROMPTS["Lung nodule"] = ("Not lung nodule", "Lung nodule")
        PROMPTS["Lung opacity"] = ("Not lung opacity", "Lung opacity")

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

    # inference
    image = loader(args.data)
    inference(model, tokenizer, image, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Perform Zero-shot', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)