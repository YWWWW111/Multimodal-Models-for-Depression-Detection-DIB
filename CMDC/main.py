from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import time
import datetime
from typing import *
from collections import Counter
from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  # Add import (if not present)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score, recall_score
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import wandb
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.metrics import matthews_corrcoef
from transformers import BertConfig, BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ==================== Simplified version: only keep inverse frequency weighting + overfitting prevention enhancement ====================

class InverseFrequencyWeightedL1Loss(nn.Module):
    """
    Inverse frequency weighted L1 loss (dynamically statistics based on current training set):
    - Each PHQ value (0~23) has independent weight: w = total / (K * count)
    - Then do "mean normalization" (ensure average weight = 1), avoid many weights truncated to one level
    - Forward uses round->long index lookup table, completely avoid float == float matching failure

    Overfitting prevention items remain unchanged:
    - Prediction mean constraint
    - Prediction standard deviation constraint
    - Prediction range constraint
    """
    def __init__(
        self,
        threshold: int = 9,
        phq_counts: Optional[Dict[float, int]] = None,
        min_w: float = 0.05,
        max_w: float = 8.0,
    ):
        super().__init__()
        self.threshold = threshold

        if phq_counts is None:
            # Unified into 0..23 count (set to 0 if missing)
            counts = [int(phq_counts.get(float(k), 0)) for k in range(24)]

        # Only count "appeared" categories
        nonzero_counts = [c for c in counts if c > 0]
        if len(nonzero_counts) == 0:
            # ...
            pass

        total_samples = int(sum(nonzero_counts))
        num_classes = int(len(nonzero_counts))

        # Inverse frequency: only calculate for appeared classes, unappeared class weight=0
        raw_weights = []
        for c in counts:
            if c > 0:
                raw_weights.append(total_samples / (num_classes * c))
            else:
                raw_weights.append(0.0)

        # Mean normalization: only do for appeared classes
        nz_weights = [w for w in raw_weights if w > 0]
        mean_w = float(np.mean(nz_weights)) if len(nz_weights) > 0 else 1.0
        norm_weights = [(w / mean_w) if w > 0 else 0.0 for w in raw_weights]

        # Truncate extreme values (only for appeared classes)
        norm_weights = [
            min(max(w, min_w), max_w) if w > 0 else 0.0
            for w in norm_weights
        ]

        self.register_buffer("weight_lut", torch.tensor(norm_weights, dtype=torch.float32))

        print("\n" + "=" * 60)
        print("Inverse frequency weight distribution (only count appeared PHQ + mean normalization)")
        print("=" * 60)
        for phq in range(24):
            if counts[phq] > 0:
                print(f"  PHQ={phq:2d}: count={counts[phq]:4d}, weight={norm_weights[phq]:.3f}")
        print("=" * 60 + "\n")

    def forward(self, predictions, targets):
        preds = predictions.view(-1)
        t = targets.view(-1)

        idx = torch.round(t).clamp(0, 23).long()
        w = self.weight_lut[idx]

        l1 = torch.abs(preds - t)
        weighted_loss = (l1 * w).mean()

        # Prediction mean constraint
        mean_penalty = torch.abs(preds.mean() - t.mean()) * 0.15

        # Prediction standard deviation constraint
        pred_std = preds.std()
        target_std = t.std()
        std_penalty = torch.relu(target_std * 0.4 - pred_std) * 0.1

        # Range constraint
        range_penalty = (torch.relu(preds - 27.0).mean() + torch.relu(-preds).mean()) * 0.5

        return weighted_loss + mean_penalty + std_penalty + range_penalty


class FocalRegressionLoss(nn.Module):
    """
    Focal regression loss: automatically give higher weight to hard samples
    """
    def __init__(self, alpha=2.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):
        l1_loss = torch.abs(predictions - targets)
        focal_weight = torch.pow(l1_loss / (l1_loss.max() + 1e-6), self.alpha)
        weighted_loss = focal_weight * l1_loss * self.beta
        return weighted_loss.mean()


def get_loss_function(loss_type='inverse_freq', threshold=9, phq_counts=None, **kwargs):
    if loss_type == 'inverse_freq':
        return InverseFrequencyWeightedL1Loss(
            threshold=threshold,
            phq_counts=phq_counts,
            min_w=0.3,
            max_w=5.0
        )
    elif loss_type == 'focal':
        return FocalRegressionLoss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'huber':  # Added Huber Loss
        return nn.HuberLoss(delta=1.0)  # delta=1.0 robust to outliers
    else:
        return nn.HuberLoss(delta=1.0)  # Default use Huber

# ==================== End modification ====================

# Ensure CUDA is available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")  # Display the name of the GPU being used
else:
    print("Warning: CUDA not available, fallback to CPU!")

# Explicitly set DEVICE to cuda:0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {DEVICE}")

from model import DIB
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob, vib_lambda=1e-4):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
        self.vib_lambda = vib_lambda # [Added]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosei")

DAIC_WOZ_DATASET_PATH = "/root/DIB_DAIC-WOZ/datasets/DAIC_WOZ_sentences.pkl"

parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--valid_batch_size", type=int, default=128)

parser.add_argument("--n_epochs", type=int, default=150)  # Increase epochs
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--model", type=str, choices=["bert-base-uncased"], default="bert-base-uncased")
parser.add_argument("--learning_rate", type=float, default=5e-5)  # Increase learning rate
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)

# VIB Lambda parameter, used to control regularization strength
parser.add_argument("--vib_lambda", type=float, default=1e-4, help="VIB Loss weight (lambda) in Eq. 6")

parser.add_argument(
    "--loss_type",
    type=str,
    choices=["inverse_freq", "l1", "mse", "focal", "huber"],
    default="huber",
    help="Loss function type"
)
parser.add_argument("--depression_threshold", type=int, default=9, help="Depression determination threshold (PHQ score)")

# Use fixed training set label distribution for weights
parser.add_argument(
    "--phq_count_source",
    type=str,
    choices=["fixed", "train"],
    default="fixed",
    help="Weight statistics source: fixed=use pre-statistics distribution, train=statistics based on current training set"
)

args = parser.parse_args()

# Fixed training set label distribution (from your current train statistics)
TRAIN_LABEL_COUNTS = {
    0: 3078, 1: 1328, 2: 1521, 3: 1173, 4: 1183, 5: 897,
    6: 526, 7: 1860, 8: 347, 9: 824, 10: 1314, 11: 840,
    12: 732, 13: 348, 14: 195, 15: 436, 16: 691, 17: 73,
    18: 336, 19: 484, 20: 428, 21: 0, 22: 0, 23: 233
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def return_unk():
    return 0


def variance_regularized_loss(preds, targets, var_weight=0.01):
    """
    Loss function with variance regularization, using a more stable way to calculate variance penalty
    """
    mse_loss = nn.MSELoss()(preds, targets)

    if torch.isnan(preds).any():
        print("Warning: Predictions contain NaN, loss calculation affected")
        return mse_loss

    batch_var = torch.var(preds.view(-1))
    var_penalty = var_weight / (batch_var + 1e-4)
    var_penalty = torch.clamp(var_penalty, 0, 10.0)
    return mse_loss + var_penalty


def prep_for_training(num_train_optimization_steps, phq_counts):
    """
    Prepare model for training
    """
    # [Modified] Pass vib_lambda args
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, 
        dropout_prob=args.dropout_prob,
        vib_lambda=args.vib_lambda
    )

    config = BertConfig.from_pretrained(args.model, num_labels=1, finetuning_task=args.dataset)

    custom_loss = get_loss_function(
        loss_type=args.loss_type,
        threshold=args.depression_threshold,
        phq_counts=phq_counts
    )

    model = DIB.from_pretrained(
        args.model,
        config=config,
        multimodal_config=multimodal_config,
        loss_function=custom_loss
    )

    model.to(DEVICE)
    return model


def clone_samples(samples: List[Dict]) -> List[Dict]:
    return [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in samples]


def standardize_modalities(samples: List[Dict], stats: Optional[Dict] = None) -> Dict:
    return {}


class DAICDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], max_seq_length: int, tokenizer: BertTokenizer):
        self.samples = samples
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        encoded = self.tokenizer.encode_plus(
            sample['text'],
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0),
            'visual': torch.tensor(sample['vision'], dtype=torch.float32),
            'acoustic': torch.tensor(sample['audio'], dtype=torch.float32),
            'label_ids': torch.tensor(sample['label'], dtype=torch.float32)
        }


def load_data(dataset_name: str) -> Tuple[Dict, str]:
    data_path = DAIC_WOZ_DATASET_PATH
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded dataset: {data_path}")
        print(f"  - Dataset contains splits: {list(data.keys())}")
        for split_name, split_data in data.items():
            print(f"    * {split_name}: {len(split_data)} samples")
        return data, data_path
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {data_path}")


def _build_phq_counts_from_train(train_samples):
    labels_int = []
    for s in train_samples:
        v = float(s["label"])
        labels_int.append(int(round(v)))
    counter = Counter(labels_int)
    return {float(k): int(counter.get(k, 0)) for k in range(24)}


def _build_phq_counts_fixed():
    return {float(k): int(TRAIN_LABEL_COUNTS.get(k, 0)) for k in range(24)}


def set_up_data_loader():
    """
    Set up DAIC-WOZ data loader (using standard train/dev split)
    """
    data, data_path = load_data(args.dataset)

    if 'train' not in data or 'dev' not in data:
        raise ValueError("Dataset must contain 'train' and 'dev' splits")

    print(f"\n{'='*20} DAIC-WOZ Data Preparation {'='*20}")

    train_samples = data['train']
    dev_samples = data['dev']
    test_samples = data.get('test', [])

    print(f"  - Training set sample count: {len(train_samples)}")
    print(f"  - Validation set sample count: {len(dev_samples)}")
    if test_samples:
        print(f"  - Test set sample count: {len(test_samples)}")

    # ===== Statistics PHQ distribution (0~23) =====
    if args.phq_count_source == "fixed":
        phq_counts = _build_phq_counts_fixed()
    else:
        phq_counts = _build_phq_counts_from_train(train_samples)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)

    train_dataset = DAICDataset(train_samples, args.max_seq_length, tokenizer)
    valid_dataset = DAICDataset(dev_samples, args.max_seq_length, tokenizer)

    num_train_optimization_steps = (
        int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step) * args.n_epochs
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False)

    return (
        train_dataloader,
        valid_dataloader,
        num_train_optimization_steps,
        phq_counts,
    )


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer: AdamW, scheduler):
    model.train()
    tr_loss = 0.0
    nb_tr_steps = 0
    max_grad_norm = 1.0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids = batch['input_ids'].to(DEVICE)
        visual = batch['visual'].to(DEVICE)
        acoustic = batch['acoustic'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        label_ids = batch['label_ids'].to(DEVICE)

        loss = model(
            input_ids,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_ids=label_ids,
        )

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()
        tr_loss += float(loss.item())
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0.0
    return avg_loss


def eval_epoch(model: nn.Module, valid_dataloader: DataLoader):
    model.eval()
    valid_loss = 0.0
    nb_valid_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader, desc="Validation")):
            input_ids = batch['input_ids'].to(DEVICE)
            visual = batch['visual'].to(DEVICE)
            acoustic = batch['acoustic'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            label_ids = batch['label_ids'].to(DEVICE)

            loss = model(
                input_ids,
                visual,
                acoustic,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_ids=label_ids,
            )
            valid_loss += float(loss.item())
            nb_valid_steps += 1
    return valid_loss / nb_valid_steps if nb_valid_steps > 0 else 0.0


def valid_epoch(model: nn.Module, valid_dataloader: DataLoader):
    """Make predictions on validation set"""
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader, desc="Validating")):
            input_ids = batch['input_ids'].to(DEVICE)
            visual = batch['visual'].to(DEVICE)
            acoustic = batch['acoustic'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            label_ids = batch['label_ids'].to(DEVICE)

            outputs = model(
                input_ids,
                visual,
                acoustic,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs

            preds.append(logits.detach().cpu().numpy())
            labels.append(label_ids.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    return preds, labels


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    metrics = {}
    metrics["mae"] = float(np.mean(np.abs(preds - labels)))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(labels, preds)))

    if preds.std() > 0 and labels.std() > 0:
        metrics["correlation"] = float(np.corrcoef(preds, labels)[0, 1])
    else:
        metrics["correlation"] = 0.0

    print(f"\n[Prediction Distribution Diagnosis]")
    print(f"  Predictions: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}, std={preds.std():.2f}")
    print(f"  Ground truth: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.2f}, std={labels.std():.2f}")
    print(f"  Predicted negative class (PHQ<9): {(preds < 9).sum()}/{len(preds)} ({(preds < 9).mean()*100:.1f}%)")
    print(f"  True negative class (PHQ<9): {(labels < 9).sum()}/{len(labels)} ({(labels < 9).mean()*100:.1f}%)")
    print(f"  Predicted positive class (PHQ≥9): {(preds >= 9).sum()}/{len(preds)} ({(preds >= 9).mean()*100:.1f}%)")
    print(f"  True positive class (PHQ≥9): {(labels >= 9).sum()}/{len(labels)} ({(labels >= 9).mean()*100:.1f}%)")

    def phq_bucket(score):
        rounded_score = int(score + 0.5)
        if rounded_score <= 4:
            return 0
        if rounded_score <= 9:
            return 1
        if rounded_score <= 14:
            return 2
        if rounded_score <= 19:
            return 3
        return 4

    def phq_bucket_3class(score):
        rounded_score = int(score + 0.5)
        if rounded_score <= 4:
            return 0
        if rounded_score <= 14:
            return 1
        return 2

    pred_categories = np.array([phq_bucket(v) for v in preds])
    true_categories = np.array([phq_bucket(v) for v in labels])

    metrics["multiclass_accuracy"] = float(np.mean(pred_categories == true_categories))
    metrics["multiclass_f1_macro"] = float(f1_score(true_categories, pred_categories, average="macro", zero_division=0))
    metrics["multiclass_f1_weighted"] = float(f1_score(true_categories, pred_categories, average="weighted", zero_division=0))

    pred_categories_3class = np.array([phq_bucket_3class(v) for v in preds])
    true_categories_3class = np.array([phq_bucket_3class(v) for v in labels])

    metrics["triclass_accuracy"] = float(np.mean(pred_categories_3class == true_categories_3class))
    metrics["triclass_f1_macro"] = float(f1_score(true_categories_3class, pred_categories_3class, average="macro", zero_division=0))
    metrics["triclass_f1_weighted"] = float(f1_score(true_categories_3class, pred_categories_3class, average="weighted", zero_division=0))

    pred_binary = preds >= 9
    true_binary = labels >= 9

    # Add confusion matrix debugging
    cm = confusion_matrix(true_binary, pred_binary)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n[Binary Classification Confusion Matrix]")
    print(f"  TN (True Negative): {tn}, FP (False Positive): {fp}")
    print(f"  FN (False Negative): {fn}, TP (True Positive): {tp}")
    print(f"  Positive class precision: {tp / (tp + fp):.4f} (TP / (TP+FP))")
    print(f"  Positive class recall: {tp / (tp + fn):.4f} (TP / (TP+FN))")
    print(f"  Negative class precision: {tn / (tn + fn):.4f} (TN / (TN+FN))")
    print(f"  Negative class recall: {tn / (tn + fp):.4f} (TN / (TN+FP))")

    metrics["binary_accuracy"] = float(accuracy_score(true_binary, pred_binary))
    metrics["binary_f1_macro"] = float(f1_score(true_binary, pred_binary, average="macro", zero_division=0))
    metrics["binary_f1_weighted"] = float(f1_score(true_binary, pred_binary, average="weighted", zero_division=0))

    f1_per_class = f1_score(true_binary, pred_binary, average=None, zero_division=0)
    metrics["binary_f1_negative"] = float(f1_per_class[0])
    metrics["binary_f1_positive"] = float(f1_per_class[1])

    metrics["binary_f1"] = metrics["binary_f1_macro"]

    # Calculate AUROC (using continuous predictions as scores)
    metrics["auroc"] = float(roc_auc_score(true_binary, preds))

    # Calculate Precision and Recall (binary classification)
    metrics["precision"] = float(precision_score(true_binary, pred_binary))
    metrics["recall"] = float(recall_score(true_binary, pred_binary))

    print(f"\n[Binary Classification Performance Details]")
    print(f"  Negative class (PHQ<9) F1: {f1_per_class[0]:.4f}")
    print(f"  Positive class (PHQ≥9) F1: {f1_per_class[1]:.4f}")
    print(f"  Macro F1 (unweighted): {metrics['binary_f1_macro']:.4f}")
    print(f"  Weighted F1 (weighted): {metrics['binary_f1_weighted']:.4f}")
    print(f"  Accuracy: {metrics['binary_accuracy']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}, AUROC: {metrics['auroc']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

    metrics["pred_categories"] = pred_categories
    metrics["true_categories"] = true_categories
    metrics["pred_categories_3class"] = pred_categories_3class
    metrics["true_categories_3class"] = true_categories_3class
    metrics["preds"] = preds
    metrics["labels"] = labels
    return metrics


def save_run_artifacts(file_prefix, args, metrics, predictions, all_epoch_details, best_metrics=None):
    os.makedirs("run_artifacts", exist_ok=True)

    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    output_summary = {
        "args": vars(args),
        "summary_metrics": convert_numpy_types(metrics),
        "best_epoch_predictions_sample": convert_numpy_types(predictions),
    }
    if best_metrics:
        output_summary["best_metrics"] = convert_numpy_types(best_metrics)

    summary_path = os.path.join("run_artifacts", f"{file_prefix}_summary.json")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(output_summary, f, indent=4, ensure_ascii=False)
        print(f"✅ Run summary saved to: {summary_path}")
    except Exception as e:
        print(f"❌ Failed to save run summary: {e}")

    details_path = os.path.join("run_artifacts", f"{file_prefix}_epoch_details.json")
    try:
        details_output = {
            "args": vars(args),
            "epoch_details": convert_numpy_types(all_epoch_details)
        }
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details_output, f, indent=4, ensure_ascii=False)
        print(f"✅ Epoch details saved to: {details_path}")
    except Exception as e:
        print(f"❌ Failed to save epoch details: {e}")


def train(model, train_dataloader, validation_dataloader, num_train_optimization_steps, fold_id, run_timestamp):
    os.makedirs("saved_models", exist_ok=True)
    model_save_path = f"saved_models/{args.dataset}_fold_{fold_id}_best_model.pt"
    file_prefix = f"run_{run_timestamp}_fold_{fold_id}"

    run_metrics = {
        "epochs": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "valid_mae": [],
        "valid_correlation": [],
        "valid_f1": [],
        "valid_acc7": [],
        "valid_multiclass_f1": [],
        "valid_triclass_accuracy": [],
        "valid_triclass_f1": [],
        "best_mae": float('inf'),
        "best_acc": 0.0,
        "best_acc_7": 0.0,
        "best_f_score": 0.0,
        "best_corr": -1.0,
        "best_multiclass_f1": 0.0,
        "best_triclass_accuracy": 0.0,
        "best_triclass_f1": 0.0,
        "best_rmse": float('inf'),
        "best_auroc": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0
    }

    valid_losses = []
    valid_accuracies = []
    all_epoch_details = []

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.bert.named_parameters()],
            "lr": args.learning_rate,
            "weight_decay": 0.02
        },
        {
            "params": [p for n, p in model.named_parameters() if "bert" not in n],
            "lr": args.learning_rate * 3,
            "weight_decay": 0.15
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    # Option 1: Fixed period (recommended)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,  # Restart every 30 epochs
        T_mult=1,  # Period remains unchanged (each time 30 epochs)
        eta_min=args.learning_rate * 0.01  # Minimum learning rate is 1% of initial
    )

    # Or option 2: Progressive period (suitable for longer training)
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=25,  # First period 25 epochs
    #     T_mult=2,  # Each restart doubles the period (25, 50, 100...)
    #     eta_min=args.learning_rate * 0.05  # Minimum learning rate is 5% of initial
    # )

    wandb.watch(model, log="all", log_freq=100)

    best_loss = float('inf')
    best_mae = float('inf')
    best_acc = 0.0
    best_acc_7 = 0.0
    best_f_score = 0.0
    best_corr = -1.0
    best_multiclass_f1 = 0.0
    best_triclass_accuracy = 0.0
    best_triclass_f1 = 0.0
    best_rmse = float('inf')
    best_auroc = 0.0
    best_precision = 0.0
    best_recall = 0.0

    patience = 100
    patience_counter = 0
    best_epoch = 0

    start_time = time.time()

    for epoch_i in range(int(args.n_epochs)):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader)

        valid_preds, valid_labels = valid_epoch(model, validation_dataloader)
        metric_dict = compute_metrics(valid_preds, valid_labels)

        current_epoch_details = {
            "epoch": epoch_i,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "statistics": metric_dict,
        }
        all_epoch_details.append(current_epoch_details)

        scalar_metrics = {
            "epoch": epoch_i,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "mae": metric_dict["mae"],
            "acc": metric_dict["binary_accuracy"],
            "acc7": metric_dict["multiclass_accuracy"],
            "acc3": metric_dict["triclass_accuracy"],
            "binary_f1": metric_dict["binary_f1"],
            "multi_f1": metric_dict["multiclass_f1_macro"],
            "tri_f1": metric_dict["triclass_f1_macro"],
            "corr": metric_dict["correlation"],
            "multiclass_f1_weighted": metric_dict["multiclass_f1_weighted"],
            "triclass_f1_weighted": metric_dict["triclass_f1_weighted"],
            "lr": scheduler.get_last_lr()[0]
        }

        epoch_time = time.time() - epoch_start_time

        print(
            "epoch:{}, train_loss:{:.4f}, valid_loss:{:.4f}, acc:{:.4f}, time:{:.2f}s".format(
                epoch_i, train_loss, valid_loss, scalar_metrics["acc"], epoch_time
            )
        )
        print(
            "current mae:{mae:.4f}, acc:{acc:.4f}, acc7:{acc7:.4f}, acc3:{acc3:.4f}, "
            "binary_f1:{binary_f1:.4f}, multi_f1:{multi_f1:.4f}, tri_f1:{tri_f1:.4f}, corr:{corr:.4f}".format(
                **scalar_metrics
            )
        )

        wandb.log(scalar_metrics)

        valid_losses.append(valid_loss)
        valid_accuracies.append(metric_dict["binary_accuracy"])

        epoch_metrics = {
            "epoch": epoch_i,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": metric_dict["binary_accuracy"],
            "valid_mae": metric_dict["mae"],
            "valid_correlation": metric_dict["correlation"],
            "valid_f1": metric_dict["binary_f1"],
            "valid_acc7": metric_dict["multiclass_accuracy"],
            "valid_multiclass_f1": metric_dict["multiclass_f1_macro"],
            "valid_triclass_accuracy": metric_dict["triclass_accuracy"],
            "valid_triclass_f1": metric_dict["triclass_f1_macro"],
            "epoch_time": epoch_time,
        }
        run_metrics["epochs"].append(epoch_metrics)

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best loss: {best_loss:.4f}, model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch_i}, best loss: {best_loss:.4f}")
                break

        if metric_dict["mae"] < best_mae:
            best_mae = metric_dict["mae"]

        if metric_dict["binary_accuracy"] > best_acc:
            best_acc = metric_dict["binary_accuracy"]

        if metric_dict["multiclass_accuracy"] > best_acc_7:
            best_acc_7 = metric_dict["multiclass_accuracy"]

        if metric_dict["triclass_accuracy"] > best_triclass_accuracy:
            best_triclass_accuracy = metric_dict["triclass_accuracy"]

        if metric_dict["binary_f1"] > best_f_score:
            best_f_score = metric_dict["binary_f1"]

        if metric_dict["correlation"] > best_corr:
            best_corr = metric_dict["correlation"]

        if metric_dict["multiclass_f1_macro"] > best_multiclass_f1:
            best_multiclass_f1 = metric_dict["multiclass_f1_macro"]

        if metric_dict["triclass_f1_macro"] > best_triclass_f1:
            best_triclass_f1 = metric_dict["triclass_f1_macro"]

        if metric_dict["rmse"] < best_rmse:
            best_rmse = metric_dict["rmse"]

        if metric_dict["auroc"] > best_auroc:
            best_auroc = metric_dict["auroc"]

        if metric_dict["precision"] > best_precision:
            best_precision = metric_dict["precision"]

        if metric_dict["recall"] > best_recall:
            best_recall = metric_dict["recall"]

    final_preds, final_labels = valid_epoch(model, validation_dataloader)
    final_metrics = compute_metrics(final_preds, final_labels)

    wandb.log({
        "final_confusion_matrix_5class": wandb.plot.confusion_matrix(
            probs=None,
            y_true=final_metrics["true_categories"],
            preds=final_metrics["pred_categories"],
            class_names=["No depression", "Mild", "Moderate", "Moderate-severe", "Severe"],
        ),
        "final_confusion_matrix_3class": wandb.plot.confusion_matrix(
            probs=None,
            y_true=final_metrics["true_categories_3class"],
            preds=final_metrics["pred_categories_3class"],
            class_names=["Normal", "Mild/Moderate", "Severe"],
        ),
        "final_predictions_histogram": wandb.Histogram(final_metrics["preds"]),
    })

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds, {total_time/60:.2f} minutes")

    run_metrics["training_time_seconds"] = float(total_time)
    run_metrics["best_epoch"] = best_epoch

    return {
        "best_mae": best_mae,
        "best_acc": best_acc,
        "best_acc_7": best_acc_7,
        "best_f_score": best_f_score,
        "best_corr": best_corr,
        "best_multiclass_f1": best_multiclass_f1,
        "best_triclass_accuracy": best_triclass_accuracy,
        "best_triclass_f1": best_triclass_f1,
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "best_auroc": best_auroc,
        "best_precision": best_precision,
        "best_recall": best_recall
    }


def main():
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        wandb.init(
            project="DIB-DAIC-WOZ",
            name=f"run_{run_timestamp}",
            config=vars(args)
        )
    except Exception as e:
        print(f"Wandb init failed: {e}")

    try:
        train_dataloader, validation_dataloader, num_train_optimization_steps, phq_counts = set_up_data_loader()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    try:
        model = prep_for_training(num_train_optimization_steps, phq_counts)
        best_metrics = train(model, train_dataloader, validation_dataloader, num_train_optimization_steps, fold_id=1, run_timestamp=run_timestamp)
        print("Training completed.")
        print(f"Best metrics: {best_metrics}")
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()
