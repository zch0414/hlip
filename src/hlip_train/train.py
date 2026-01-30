import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import json
import logging
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.precision import get_autocast

from hlip_train.zero_shot import zero_shot_eval


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    # if args.distill:
    #     dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // (args.accum_freq * args.accum_batch)
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    
    # Gradient accum in the original repo.
    if args.accum_freq > 1:
        accum_images, accum_sentences, accum_reports, accum_features = [], [], [], {} 
    # In this repo, we perform batch accum by default.
    images, sentences, reports = [], [], []

    losses = {}
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // (args.accum_freq * args.accum_batch)
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images.append(batch['image'].to(device, dtype=input_dtype, non_blocking=True))
        sentences.append(batch['sentence'].to(device=device, non_blocking=True))
        reports.append(batch['report'].to(device=device, non_blocking=True))

        if ((i + 1) % args.accum_batch) > 0:
            continue

        images = torch.cat(images, dim=0)
        sentences = torch.cat(sentences, dim=0)
        reports = torch.cat(reports, dim=0)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(image=images)
                # if args.distill:
                    #     with torch.no_grad():
                    #         dist_model_out = dist_model(images, texts)
                    #     model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                image_features = model_out.pop("image_features")
                model_out.pop("text_features")

                logit_scale_sentence = model_out.pop("logit_scale")
                model_out["image_features_sentence"] = image_features[:, 0, :].contiguous()
                if image_features.shape[1] == 1:
                    logit_scale_report = None
                    model_out["text_features_sentence"] = model(text=sentences).pop("text_features")
                elif image_features.shape[1] == 2:
                    logit_scale_report = model_out.pop("logit_bias").exp() # FIXME: currently use logit_bias hack the logit_scale for report
                    model_out["image_features_report"] = image_features[:, 1, :].contiguous()
                    text_features_sentence, text_features_report = model(text=torch.cat([sentences, reports])).pop("text_features").chunk(2, dim=0)
                    model_out["text_features_sentence"] = text_features_sentence.contiguous()
                    model_out["text_features_report"] = text_features_report.contiguous()
                    
                model_out_sentence = {
                    "image_features": model_out.pop("image_features_sentence"),
                    "text_features": model_out.pop("text_features_sentence"),
                    "logit_scale": logit_scale_sentence,
                }
                model_out_report = {
                    "image_features": model_out.pop("image_features_report", None),
                    "text_features": model_out.pop("text_features_report", None),
                    "logit_scale": logit_scale_report,
                }

                model_out_sentence.update(model_out) # NOTE: in case anything is left in the model_out but necessary for loss
                losses_sentence = loss(**model_out_sentence, output_dict=True)
                total_loss = sum(losses_sentence.values())
                losses["loss_sentence"] = total_loss

                if model_out_report["image_features"] is not None and model_out_report["text_features"] is not None and model_out_report["logit_scale"] is not None:
                    model_out_report.update(model_out) # NOTE: in case anything is left in the model_out but necessary for loss
                    losses_report = loss(**model_out_report, output_dict=True)
                    loss_report = sum(losses_report.values())
                    total_loss = 0.5 * total_loss + 0.5 * loss_report
                    losses["loss_report"] = loss_report

                losses["loss"] = total_loss
            backward(total_loss, scaler)
            images, sentences, reports = [], [], [] # reset batch accum
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(image=images)
                    image_features = model_out.pop("image_features")
                    model_out.pop("text_features")

                    model_out["image_features_sentence"] = image_features[:, 0, :]
                    if image_features.shape[1] == 1:
                        model_out["text_features_sentence"] = model(text=sentences).pop("text_features")
                    elif image_features.shape[1] == 2:
                        model_out["image_features_report"] = image_features[:, 1, :]
                        text_features_sentence, text_features_report = model(text=torch.cat([sentences, reports])).pop("text_features").chunk(2, dim=0)
                        model_out["text_features_sentence"] = text_features_sentence
                        model_out["text_features_report"] = text_features_report

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_sentences.append(sentences)
                accum_reports.append(reports)

            images, sentences, reports = [], [], [] # reset batch accum

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % (args.accum_batch * args.accum_freq)) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                sentences = accum_sentences[j]
                reports = accum_reports[j]
                with autocast():
                    model_out = model(image=images)
                    image_features = model_out.pop("image_features")
                    model_out.pop("text_features")
                    
                    logit_scale_sentence = model_out.pop("logit_scale")
                    model_out["image_features_sentence"] = image_features[:, 0, :]
                    if image_features.shape[1] == 1:
                        logit_scale_report = None
                        model_out["text_features_sentence"] = model(text=sentences).pop("text_features")
                    elif image_features.shape[1] == 2:
                        logit_scale_report = model_out.pop("logit_bias").exp() # FIXME: currently use logit_bias hack the logit_scale for report
                        model_out["image_features_report"] = image_features[:, 1, :]
                        text_features_sentence, text_features_report = model(text=torch.cat([sentences, reports])).pop("text_features").chunk(2, dim=0)
                        model_out["text_features_sentence"] = text_features_sentence
                        model_out["text_features_report"] = text_features_report

                    inputs_no_accum = {}
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    inputs_sentence = {
                        "image_features": inputs.pop("image_features_sentence"),
                        "text_features": inputs.pop("text_features_sentence"),
                        "logit_scale": logit_scale_sentence
                    }
                    inputs_report = {
                        "image_features": inputs.pop("image_features_report", None),
                        "text_features": inputs.pop("text_features_report", None),
                        "logit_scale": logit_scale_report
                    }

                    inputs_sentence.update(inputs) # NOTE: in case anything is left in the inputs but necessary for loss
                    losses_sentence = loss(**inputs_sentence, **inputs_no_accum, output_dict=True)
                    
                    del inputs_sentence
                    total_loss = sum(losses_sentence.values())
                    losses["loss_sentence"] = total_loss

                    if inputs_report["image_features"] is not None and inputs_report["text_features"] is not None and inputs_report["logit_scale"] is not None:
                        inputs_report.update(inputs) # NOTE: in case anything is left in the inputs but necessary for loss
                        losses_report = loss(**inputs_report, output_dict=True)
                        
                        del inputs_report
                        loss_report = sum(losses_report.values())
                        total_loss = 0.5 * total_loss + 0.5 * loss_report
                        losses["loss_report"] = loss_report

                    losses["loss"] = total_loss

                    del inputs
                    del inputs_no_accum

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Reset gradient accum as in the original repo.
        if args.accum_freq > 1:
            accum_images, accum_sentences, accum_reports, accum_features = [], [], [], {}
        images, sentences, reports = [], [], [] # reset batch accum for this repo.

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            m = unwrap_model(model)
            m.logit_scale.clamp_(0, math.log(100))
            # NOTE: we also clamp logit_bias
            if getattr(m, "logit_bias", None) is not None:
                m.logit_bias.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # NOTE batch_size = len(images) in the original repo.
            # In this repo, we compute the num_samples with batch_size in arguments.
            num_samples = batch_count * args.batch_size * args.accum_batch * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), args.batch_size * args.accum_batch)

            logit_scale_scalar_sentence = logit_scale_sentence.item()
            logit_scale_scalar_report = logit_scale_report.item() if logit_scale_report is not None else 0.0
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_batch * args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_batch * args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale (Sentence): {logit_scale_scalar_sentence:.3f} "
                f"Logit Scale (Report): {logit_scale_scalar_report:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale (sentence)": logit_scale_scalar_sentence,
                "scale (report)": logit_scale_scalar_report,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    model.eval()

    # Run distributed zero-shot evaluation first
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)

    # Run evaluation on rank 0
    if not is_master(args):
        return {}
    
    metrics = {}
    metrics.update(zero_shot_metrics)

    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'valid' in data and (args.valid_frequency and ((epoch % args.valid_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['valid'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        # NOTE original repo compute the clip loss on eval datasets
        # here we compute clip score instead
        cumulative_clip_score = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images = batch['image'].to(device=device, dtype=input_dtype, non_blocking=True)
                texts = batch['report'].to(device=device, non_blocking=True)
                
                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"][:, -1, :]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_bias"] if "logit_bias" in model_out else model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    # all_image_features.append(image_features.cpu())
                    # all_text_features.append(text_features.cpu())
                    # NOTE we only use a small validation set (~1000)
                    # system RAM is more sensitive than GPU memory in our case
                    # so we do not compute features on CPU
                    all_image_features.append(image_features)
                    all_text_features.append(text_features)
                    logit_scale = logit_scale.mean()

                    # NOTE batch_size = images.shape[0] in the original repo
                    # here we use image_features.shape[0] instead
                    batch_size = image_features.shape[0]                    
                    clip_scores_per_image = torch.clamp(image_features @ text_features.t(), min=0) * 100
                    total_clip_scores = clip_scores_per_image.trace()

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_clip_score += total_clip_scores
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Score: {cumulative_clip_score / num_samples:.6f}\t"
                    )

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            valid_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                # logit_scale=logit_scale.cpu(),
                logit_scale=logit_scale,
            )
            clip_score = cumulative_clip_score / num_samples
            metrics.update(
                {**valid_metrics, "valid_clip_score":clip_score.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"valid_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"valid/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // (args.accum_freq * args.accum_batch)
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
