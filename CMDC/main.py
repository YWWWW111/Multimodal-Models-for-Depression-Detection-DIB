from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import pickle
import random
import sys
import time
import datetime

from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import pdist, squareform

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.metrics import matthews_corrcoef
from transformers import BertConfig, BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

import wandb

# 确保CUDA可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空GPU缓存
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")  # 显示正在使用的GPU名称
else:
    print("警告: CUDA不可用，回退到CPU!")

# 显式设置DEVICE为cuda:0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"设备设置为: {DEVICE}")

from model import DIB
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM

# --- 新增 MultimodalConfig 和 BertConfig 导入 ---
from transformers import BertConfig

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
# --- 结束新增 ---

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["mosi", "mosei"], default="mosei")

# ============= 修改点 1: 更新数据集路径 =============
MOSEI_DATASET_ABS_PATH = "/root/DIB_CMDC_TAV_SL_Fusion/datasets/CMDC_Text_CV_SL.pkl"

# --- 修改点 2: 设置 max_seq_length ---
parser.add_argument("--max_seq_length", type=int, default=400)
parser.add_argument("--num_text_lines", type=int, default=12,
                    help="每个样本包含的独立文本行数（例如12个问题对应的回答），默认为12")

# 减小 train_batch_size 以增加每个epoch的批次数
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--dev_batch_size", type=int, default=4)
parser.add_argument("--valid_batch_size", type=int, default=4)  # 改名为 valid_batch_size
parser.add_argument("--n_epochs", type=int, default=60)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.3)
parser.add_argument(
    "--model",
    type=str,
    choices=["bert-base-uncased"],
    default="bert-base-uncased",
)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

# --- 新增 set_random_seed 函数定义 ---
def set_random_seed(seed):
    """
    设置随机种子以确保结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# --- 结束新增 ---

def return_unk():
    return 0


def variance_regularized_loss(preds, targets, var_weight=0.01):
    """
    带有方差正则化的损失函数，使用更稳定的方式计算方差惩罚
    """
    mse_loss = nn.MSELoss()(preds, targets)
    
    # 检查预测值是否有NaN
    if torch.isnan(preds).any():
        print("警告：预测值包含NaN，损失计算受影响")
        return mse_loss
    
    # 计算预测值的方差
    batch_var = torch.var(preds.view(-1))
    
    # 使用更稳定的方式计算方差惩罚
    # 避免对数，改用直接除法
    var_penalty = var_weight / (batch_var + 1e-4)
    
    # 限制方差惩罚的大小，防止数值爆炸
    var_penalty = torch.clamp(var_penalty, 0, 10.0)
    
    return mse_loss + var_penalty

# ---  prep_for_training 函数定义 ---
def prep_for_training(num_train_optimization_steps):
    """
    准备模型以进行训练
    """
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )

    # 加载BERT配置，并指定回归任务（num_labels=1）
    config = BertConfig.from_pretrained(
        args.model, num_labels=1, finetuning_task=args.dataset
    )

    # 从预训练的BERT模型初始化我们的DIB模型
    model = DIB.from_pretrained(
        args.model,
        config=config,
        multimodal_config=multimodal_config,
    )

    model.to(DEVICE)

    return model
# --- 结束新增 ---

def clone_samples(samples: List[Dict]) -> List[Dict]:
    """深拷贝样本列表以避免修改原始数据"""
    return [
        {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
        for s in samples
    ]

def standardize_modalities(samples: List[Dict], stats: Optional[Dict] = None) -> Dict:
    """保留原始音频/视觉特征，不做全局归一化"""
    return {}

class CMDCDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], max_seq_length: int, tokenizer: BertTokenizer, num_text_lines: int):
        self.samples = samples
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.num_text_lines = num_text_lines

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        
        # 获取文本数据，预期是一个包含12个字符串的列表
        text_list = sample["text"]
        
        # 健壮性检查：确保是列表
        if isinstance(text_list, np.ndarray):
            text_list = text_list.tolist()
        if isinstance(text_list, str):
            text_list = [text_list]

        # 1. 准备数据：确保列表长度为 num_text_lines (12)
        # 计算有效的行数
        valid_len = len(text_list)
        
        # 构造 line_mask (标记哪些行是真实数据)
        line_mask = [1] * min(valid_len, self.num_text_lines) + [0] * max(0, self.num_text_lines - valid_len)
        
        # 填充文本列表到固定长度 12
        if valid_len < self.num_text_lines:
            padded_text_list = text_list + [""] * (self.num_text_lines - valid_len)
        else:
            padded_text_list = text_list[:self.num_text_lines]

        # 2. 批量 Tokenize
        encoded = self.tokenizer(
            padded_text_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
            add_special_tokens=True
        )

        # 获取结果 Tensor
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        line_mask_tensor = torch.tensor(line_mask, dtype=torch.float)

        # 处理视听特征
        visual_vec = torch.from_numpy(sample["vision"]).float()
        acoustic_vec = torch.from_numpy(sample["audio"]).float()
        line_mask_tensor = torch.tensor(line_mask, dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.float32)

        return input_ids, visual_vec, acoustic_vec, attention_mask, token_type_ids, line_mask_tensor, label

def load_data(dataset_name: str) -> Tuple[Dict, str]:
    """
    加载CMDC_Text_CV_SL.pkl数据集（由CMDC_Manual_Split.py生成）
    数据格式: {fold_name: {'train': [...], 'val': [...]}}
    注意：此数据集只有train和val，没有独立的test集
    """
    data_path = MOSEI_DATASET_ABS_PATH
    
    try:
        print(f"尝试加载指定数据集: {data_path}")
        with open(data_path, "rb") as handle:
            data = pickle.load(handle)
        print(f"✅ 成功从 {data_path} 加载数据集")
        return data, data_path
    except FileNotFoundError:
        print(f"❌ 致命错误: 找不到指定的数据集: {data_path}")
        print("请确保该文件存在于正确的位置。程序将终止。")
        raise

def set_up_data_loader(fold_id: int):
    """
    为指定的折（fold）设置数据加载器。
    """
    data, data_path = load_data(args.dataset)
    
    current_fold_key = f'fold{fold_id}'

    if current_fold_key not in data:
        raise ValueError(f"数据集 {data_path} 缺少 {current_fold_key}，请检查数据预处理脚本。")

    print(f"\n{'='*20} FOLD {fold_id} {'='*20}")
    print(f"为 FOLD {fold_id} 准备数据加载器...")
    
    # ✅ 使用CMDC_Manual_Split生成的数据：只有train和val
    if 'train' not in data[current_fold_key] or 'val' not in data[current_fold_key]:
         raise ValueError(f"{current_fold_key} 格式错误，必须包含 'train' 和 'val' 键")

    train_samples_raw = data[current_fold_key]['train']
    val_samples_raw = data[current_fold_key]['val']
    
    print(f"  - 训练集样本数: {len(train_samples_raw)}")
    print(f"  - 验证集样本数: {len(val_samples_raw)}")

    # 直接使用预处理好的train和val数据
    train_samples = clone_samples(train_samples_raw)
    val_samples = clone_samples(val_samples_raw)

    # 对训练集计算标准化参数，并应用到验证集
    print("  - 正在对数据进行标准化...")
    stats = standardize_modalities(train_samples)
    standardize_modalities(val_samples, stats)
    print("  - 标准化完成。")

    # 初始化Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=True)

    train_dataset = CMDCDataset(train_samples, args.max_seq_length, tokenizer, args.num_text_lines)
    valid_dataset = CMDCDataset(val_samples, args.max_seq_length, tokenizer, args.num_text_lines)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )
    print(f"  - 训练样本数: {len(train_dataset)}")
    print(f"  - 验证样本数: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, shuffle=False
    )

    # 检查数据结构
    if len(train_dataset) > 0:
        sample = train_samples[0]
        print(f"\n=== 训练样本检查 (FOLD {fold_id}) ===")
        
        text_val = sample['text']
        if isinstance(text_val, list):
            text_info = f"list of {len(text_val)} items"
        elif isinstance(text_val, np.ndarray):
            text_info = f"ndarray shape {text_val.shape}"
        elif isinstance(text_val, str):
            text_info = f"string length {len(text_val)}"
        else:
            text_info = f"unknown type {type(text_val)}"
            
        print(f"text info: {text_info}, audio 形状: {sample['audio'].shape}, vision 形状: {sample['vision'].shape}, label: {sample['label']}")
    
    return (
        train_dataloader,
        valid_dataloader,
        num_train_optimization_steps,
    )

def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer: AdamW, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    
    max_grad_norm = 1.0
    
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        # 直接解包新的数据结构，chunk_mask 改名为 line_mask
        input_ids, visual, acoustic, attention_mask, token_type_ids, line_mask, label_ids = [t.to(DEVICE) for t in batch]

        # --- 开始调试代码 ---
        # 只在第一个批次打印，避免刷屏
        if step == 0:
            print(f"\n[DEBUG] 批次 {step} 数据形状:")
            print(f"  - input_ids: {input_ids.shape}")
            print(f"  - visual: {visual.shape}")
            print(f"  - acoustic: {acoustic.shape}")
            print(f"  - attention_mask: {attention_mask.shape}")
            print(f"  - token_type_ids: {token_type_ids.shape}")
            print(f"  - line_mask: {line_mask.shape}")
            print(f"  - label_ids: {label_ids.shape}")
            
            # 检查数据范围
            print(f"\n[DEBUG] 数据范围检查:")
            print(f"  - visual min/max: {visual.min().item():.4f} / {visual.max().item():.4f}")
            print(f"  - acoustic min/max: {acoustic.min().item():.4f} / {acoustic.max().item():.4f}")
            print(f"  - label_ids min/max: {label_ids.min().item():.4f} / {label_ids.max().item():.4f}")
            
            # 检查NaN
            print(f"\n[DEBUG] NaN检查:")
            print(f"  - visual contains NaN: {torch.isnan(visual).any().item()}")
            print(f"  - acoustic contains NaN: {torch.isnan(acoustic).any().item()}")
            print(f"  - label_ids contains NaN: {torch.isnan(label_ids).any().item()}")
            
            # 检查line_mask
            print(f"\n[DEBUG] line_mask检查:")
            print(f"  - line_mask示例: {line_mask[0].cpu().numpy()}")
            print(f"  - 每个样本的有效行数: {line_mask.sum(dim=1).cpu().numpy()}")
        # --- 结束调试代码 ---

        # 数据检查并清理
        visual = torch.nan_to_num(visual)
        acoustic = torch.nan_to_num(acoustic)
            
        # 清除之前的梯度 (使用传入的 optimizer)
        optimizer.zero_grad()
        
        try:
            # 模型前向传播
            loss = model(
                input_ids=input_ids,
                visual=visual,
                acoustic=acoustic,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                line_mask=line_mask,
                label_ids=label_ids
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            
        except RuntimeError as e:
            print(f"训练步骤 {step} 出现运行时错误: {e}")
            print(f"跳过此批次...")
            continue
    
    return tr_loss / nb_tr_steps if nb_tr_steps > 0 else 0

def eval_epoch(model: nn.Module, valid_dataloader: DataLoader):
    model.eval()
    valid_loss = 0
    nb_valid_examples, nb_valid_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader, desc="Validation")):
            input_ids, visual, acoustic, attention_mask, token_type_ids, line_mask, label_ids = [t.to(DEVICE) for t in batch]
            
            visual = torch.nan_to_num(visual)
            acoustic = torch.nan_to_num(acoustic)
            
            loss = model(
                input_ids=input_ids,
                visual=visual,
                acoustic=acoustic,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                line_mask=line_mask,
                label_ids=label_ids
            )
            
            valid_loss += loss.item()
            nb_valid_examples += input_ids.size(0)
            nb_valid_steps += 1

    return valid_loss / nb_valid_steps if nb_valid_steps > 0 else 0


def valid_epoch(model: nn.Module, valid_dataloader: DataLoader):
    """在验证集上进行预测"""
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Validation Prediction"):
            input_ids, visual, acoustic, attention_mask, token_type_ids, line_mask, label_ids = [t.to(DEVICE) for t in batch]
            
            visual = torch.nan_to_num(visual)
            acoustic = torch.nan_to_num(acoustic)
            
            logits = model.test(
                input_ids=input_ids,
                visual=visual,
                acoustic=acoustic,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                line_mask=line_mask
            )
            
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            
            preds.extend(logits.flatten().tolist())
            labels.extend(label_ids.flatten().tolist())

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """统一计算各项评估指标并返回结果"""
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    metrics = {}
    metrics["mae"] = float(np.mean(np.abs(preds - labels)))

    if preds.std() > 0 and labels.std() > 0:
        metrics["correlation"] = float(np.corrcoef(preds, labels)[0, 1])
    else:
        metrics["correlation"] = 0.0

    # ===== 新增：预测分布诊断 =====
    print(f"\n[预测分布诊断]")
    print(f"  预测值: min={preds.min():.2f}, max={preds.max():.2f}, mean={preds.mean():.2f}, std={preds.std():.2f}")
    print(f"  真实值: min={labels.min():.2f}, max={labels.max():.2f}, mean={labels.mean():.2f}, std={labels.std():.2f}")
    print(f"  预测为负类(PHQ<9): {(preds < 9).sum()}/{len(preds)} ({(preds < 9).mean()*100:.1f}%)")
    print(f"  真实为负类(PHQ<9): {(labels < 9).sum()}/{len(labels)} ({(labels < 9).mean()*100:.1f}%)")

    # 五分类函数
    def phq_bucket(score):
        rounded_score = int(score + 0.5)
        if rounded_score <= 4:
            return 0      # 0-4: 无抑郁
        if rounded_score <= 9:
            return 1      # 5-9: 轻度
        if rounded_score <= 14:
            return 2      # 10-14: 中度
        if rounded_score <= 19:
            return 3      # 15-19: 中重度
        return 4      # 20及以上: 重度

    # 三分类函数
    def phq_bucket_3class(score):
        rounded_score = int(score + 0.5)
        if rounded_score <= 4:
            return 0      # 0-4: 正常
        if rounded_score <= 14:
            return 1      # 5-14: 轻/中度
        return 2          # 15-27: 重度

    # 五分类指标
    pred_categories = np.array([phq_bucket(v) for v in preds])
    true_categories = np.array([phq_bucket(v) for v in labels])

    metrics["multiclass_accuracy"] = float(np.mean(pred_categories == true_categories))
    metrics["multiclass_f1_macro"] = float(
        f1_score(true_categories, pred_categories, average="macro", zero_division=0)
    )
    metrics["multiclass_f1_weighted"] = float(
        f1_score(true_categories, pred_categories, average="weighted", zero_division=0)
    )

    # 三分类指标
    pred_categories_3class = np.array([phq_bucket_3class(v) for v in preds])
    true_categories_3class = np.array([phq_bucket_3class(v) for v in labels])

    metrics["triclass_accuracy"] = float(np.mean(pred_categories_3class == true_categories_3class))
    metrics["triclass_f1_macro"] = float(
        f1_score(true_categories_3class, pred_categories_3class, average="macro", zero_division=0)
    )
    metrics["triclass_f1_weighted"] = float(
        f1_score(true_categories_3class, pred_categories_3class, average="weighted", zero_division=0)
    )

    # ===== 修改：二分类指标 - 改用macro平均 =====
    pred_binary = preds >= 9
    true_binary = labels >= 9
    
    metrics["binary_accuracy"] = float(accuracy_score(true_binary, pred_binary))
    
    # 🔥 关键修改：改用 macro 平均
    metrics["binary_f1_macro"] = float(
        f1_score(true_binary, pred_binary, average="macro", zero_division=0)
    )
    
    # 保留 weighted 用于对比
    metrics["binary_f1_weighted"] = float(
        f1_score(true_binary, pred_binary, average="weighted", zero_division=0)
    )
    
    # 计算每个类别的F1（用于详细分析）
    f1_per_class = f1_score(true_binary, pred_binary, average=None, zero_division=0)
    metrics["binary_f1_negative"] = float(f1_per_class[0])  # PHQ<9
    metrics["binary_f1_positive"] = float(f1_per_class[1])  # PHQ≥9
    
    # 🔥 macro
    metrics["binary_f1"] = metrics["binary_f1_macro"]
    
    # 打印详细报告
    print(f"\n[二分类性能详情]")
    print(f"  负类(PHQ<9) F1: {f1_per_class[0]:.4f}")
    print(f"  正类(PHQ≥9) F1: {f1_per_class[1]:.4f}")
    print(f"  Macro F1 (不加权): {metrics['binary_f1_macro']:.4f}")
    print(f"  Weighted F1 (加权): {metrics['binary_f1_weighted']:.4f}")
    print(f"  准确率: {metrics['binary_accuracy']:.4f}")

    metrics["pred_categories"] = pred_categories
    metrics["true_categories"] = true_categories
    metrics["pred_categories_3class"] = pred_categories_3class
    metrics["true_categories_3class"] = true_categories_3class
    metrics["preds"] = preds
    metrics["labels"] = labels
    return metrics

# 2. 定义 save_run_artifacts 函数
def save_run_artifacts(file_prefix, args, metrics, predictions, all_epoch_details):
    """将运行的产物（参数、指标、预测、各epoch详情）保存到JSON文件"""
    # 创建保存产物的目录
    os.makedirs("run_artifacts", exist_ok=True)

    # 辅助函数，用于将numpy类型转换为JSON兼容类型
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

    # 准备要保存的数据
    output_summary = {
        "args": vars(args),
        "summary_metrics": convert_numpy_types(metrics),
        "best_epoch_predictions_sample": convert_numpy_types(predictions),
    }
    
    # 保存总体摘要文件
    summary_path = os.path.join("run_artifacts", f"{file_prefix}_summary.json")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(output_summary, f, indent=2, ensure_ascii=False)
        print(f"✅ 运行摘要已保存到: {summary_path}")
    except Exception as e:
        print(f"❌ 保存运行摘要失败: {e}")

    # 保存每个epoch的详细数据
    details_path = os.path.join("run_artifacts", f"{file_prefix}_epoch_details.json")
    try:
        details_output = {
            "args": vars(args),
            "epoch_details": convert_numpy_types(all_epoch_details)
        }
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details_output, f, indent=2, ensure_ascii=False)
        print(f"✅ 每个Epoch的详细数据已保存到: {details_path}")
    except Exception as e:
        print(f"❌ 保存Epoch详细数据失败: {e}")


def train(model, train_dataloader, validation_dataloader, num_train_optimization_steps, fold_id, run_timestamp):
    # 创建保存模型的目录
    os.makedirs("saved_models", exist_ok=True)
    # 修改模型保存路径以包含 fold_id
    model_save_path = f"saved_models/{args.dataset}_fold_{fold_id}_best_model.pt"
    
    # 使用传入的时间戳和 fold_id 构建唯一文件前缀
    file_prefix = f"run_{run_timestamp}_fold_{fold_id}"
    
    # 初始化用于收集指标的字典
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
        "valid_triclass_accuracy": [],      # 新增
        "valid_triclass_f1": [],            # 新增
        "best_mae": float('inf'),
        "best_acc": 0.0,
        "best_acc_7": 0.0,
        "best_f_score": 0.0,
        "best_corr": -1.0,
        "best_multiclass_f1": 0.0,
        "best_triclass_accuracy": 0.0,      # 新增
        "best_triclass_f1": 0.0             # 新增
    }
    
    # 初始化 valid_losses 和 valid_accuracies 列表
    valid_losses = []
    valid_accuracies = []
    
    # 初始化用于收集预测结果的列表
    predictions_data = []
    
    # 新增：初始化用于收集每个epoch详细信息的列表
    all_epoch_details = []
    
    # 在这里创建优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, eps=1e-8)
    warmup_steps = int(0.1 * num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=num_train_optimization_steps
    )
    
    # --- W&B ---
    # 使用 wandb.watch() 追踪模型梯度和参数
    wandb.watch(model, log="all", log_freq=100)
    
    # ✅ 修复：添加三分类的最佳指标
    best_loss = float('inf')
    best_mae = float('inf')
    best_acc = 0.0
    best_acc_7 = 0.0
    best_f_score = 0.0
    best_corr = -1.0
    best_multiclass_f1 = 0.0
    best_triclass_accuracy = 0.0  # 新增
    best_triclass_f1 = 0.0         # 新增
    
    # ✅ 新增：早停机制
    patience = 15
    patience_counter = 0
    best_epoch = 0
    
    # 记录训练开始时间
    start_time = time.time()
    
    for epoch_i in range(int(args.n_epochs)):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader)
        
        # ✅ 在验证集上计算指标（用于模型选择和早停）
        valid_preds, valid_labels = valid_epoch(model, validation_dataloader)
        metric_dict = compute_metrics(valid_preds, valid_labels)

        # 新增：收集当前epoch的详细信息
        current_epoch_details = {
            "epoch": epoch_i,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "statistics": metric_dict,
        }
        all_epoch_details.append(current_epoch_details)

        # ✅ 修复：添加三分类指标到 scalar_metrics
        scalar_metrics = {
            "epoch": epoch_i,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "mae": metric_dict["mae"],
            "acc": metric_dict["binary_accuracy"],
            "acc7": metric_dict["multiclass_accuracy"],
            "acc3": metric_dict["triclass_accuracy"],           # 新增
            "binary_f1": metric_dict["binary_f1"],
            "multi_f1": metric_dict["multiclass_f1_macro"],
            "tri_f1": metric_dict["triclass_f1_macro"],         # 新增
            "corr": metric_dict["correlation"],
            "multiclass_f1_weighted": metric_dict["multiclass_f1_weighted"],
            "triclass_f1_weighted": metric_dict["triclass_f1_weighted"],  # 新增
            "lr": scheduler.get_last_lr()[0]
        }

        epoch_time = time.time() - epoch_start_time

        print(
            "epoch:{}, train_loss:{:.4f}, valid_loss:{:.4f}, acc:{:.4f}, time:{:.2f}s".format(
                epoch_i, train_loss, valid_loss, scalar_metrics["acc"], epoch_time
            )
        )
        print(
            "current mae:{mae:.4f}, acc:{acc:.4f}, acc7:{acc7:.4f}, acc3:{acc3:.4f}, binary_f1:{binary_f1:.4f}, multi_f1:{multi_f1:.4f}, tri_f1:{tri_f1:.4f}, corr:{corr:.4f}".format(
                **scalar_metrics
            )
        )

        # --- W&B 日志记录 (只记录标量) ---
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
            "valid_triclass_accuracy": metric_dict["triclass_accuracy"],        # 新增
            "valid_triclass_f1": metric_dict["triclass_f1_macro"],              # 新增
            "epoch_time": epoch_time,
        }
        
        # 添加到指标记录中
        run_metrics["epochs"].append(epoch_metrics)
        
        # ✅ 修复：早停机制
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            best_epoch = epoch_i
            
            torch.save(model.state_dict(), model_save_path)
            print(f"🔥 新的最佳验证损失: {best_loss:.4f}，模型已保存到 {model_save_path}")
            
            # ✅ 修复：添加三分类指标到 predictions_data
            predictions_data = {
                "epoch": epoch_i,
                "predictions": valid_preds.tolist(),
                "labels": valid_labels.tolist(),
                "mae": metric_dict["mae"],
                "acc": metric_dict["binary_accuracy"],
                "acc7": metric_dict["multiclass_accuracy"],
                "acc3": metric_dict["triclass_accuracy"],                # 新增
                "binary_f1": metric_dict["binary_f1"],
                "multi_f1": metric_dict["multiclass_f1_macro"],
                "tri_f1": metric_dict["triclass_f1_macro"],              # 新增
                "corr": metric_dict["correlation"]
            }
            
            save_run_artifacts(file_prefix, args, run_metrics, predictions_data, all_epoch_details)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ 早停：验证损失在 {patience} 个epoch内未改善（最佳epoch: {best_epoch}）")
                break

        # 2. 独立检查并更新其他各项最佳指标
        # MAE: 越小越好
        if metric_dict["mae"] < best_mae:
            best_mae = metric_dict["mae"]
            print(f"  ✨ 新的最佳MAE: {best_mae:.4f}")
        
        # Accuracy: 越大越好
        if metric_dict["binary_accuracy"] > best_acc:
            best_acc = metric_dict["binary_accuracy"]
            print(f"  ✨ 新的最佳二分类准确率: {best_acc:.4f}")
        
        # 7-class Accuracy: 越大越好
        if metric_dict["multiclass_accuracy"] > best_acc_7:
            best_acc_7 = metric_dict["multiclass_accuracy"]
            print(f"  ✨ 新的最佳五分类准确率: {best_acc_7:.4f}")
        
        # ✅ 新增：三分类最佳指标追踪
        if metric_dict["triclass_accuracy"] > best_triclass_accuracy:
            best_triclass_accuracy = metric_dict["triclass_accuracy"]
            print(f"  ✨ 新的最佳三分类准确率: {best_triclass_accuracy:.4f}")
        
        if metric_dict["binary_f1"] > best_f_score:
            best_f_score = metric_dict["binary_f1"]
            print(f"  ✨ 新的最佳F1分数: {best_f_score:.4f}")
        
        # Correlation: 越大越好
        if metric_dict["correlation"] > best_corr:
            best_corr = metric_dict["correlation"]
            print(f"  ✨ 新的最佳相关系数: {best_corr:.4f}")
        
        # Multiclass F1: 越大越好
        if metric_dict["multiclass_f1_macro"] > best_multiclass_f1:
            best_multiclass_f1 = metric_dict["multiclass_f1_macro"]
            print(f"  ✨ 新的最佳五分类F1: {best_multiclass_f1:.4f}")
        
        # ✅ 新增：三分类F1最佳指标追踪
        if metric_dict["triclass_f1_macro"] > best_triclass_f1:
            best_triclass_f1 = metric_dict["triclass_f1_macro"]
            print(f"  ✨ 新的最佳三分类F1: {best_triclass_f1:.4f}")
    
    # ✅ 修复：更新 WandB 混淆矩阵
    final_preds, final_labels = valid_epoch(model, validation_dataloader)
    final_metrics = compute_metrics(final_preds, final_labels)  
    
    wandb.log({
        "final_confusion_matrix_5class": wandb.plot.confusion_matrix(
            probs=None,
            y_true=final_metrics["true_categories"],
            preds=final_metrics["pred_categories"],
            class_names=["无抑郁", "轻度", "中度", "中重度", "重度"],
        ),
        "final_confusion_matrix_3class": wandb.plot.confusion_matrix(
            probs=None,
            y_true=final_metrics["true_categories_3class"],
            preds=final_metrics["pred_categories_3class"],
            class_names=["正常", "轻/中度", "重度"],
        ),
        "final_predictions_histogram": wandb.Histogram(final_metrics["preds"]),
    })

    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}秒, {total_time/60:.2f}分钟")
    
    run_metrics["training_time_seconds"] = float(total_time)
    run_metrics["best_epoch"] = best_epoch  # 新增
    
    # ✅ 修复：返回值包含三分类指标
    return {
        "best_mae": best_mae,
        "best_acc": best_acc,
        "best_acc_7": best_acc_7,
        "best_f_score": best_f_score,
        "best_corr": best_corr,
        "best_multiclass_f1": best_multiclass_f1,
        "best_triclass_accuracy": best_triclass_accuracy,    # 新增
        "best_triclass_f1": best_triclass_f1,                # 新增
        "best_epoch": best_epoch                              # 新增
    }

def main():
    # ✅ 修复：完整的随机种子设置
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        wandb.login()
        wandb.init(project="CMDC_Depression_Detection", name=f"run_{run_timestamp}", config=vars(args))
        
        # ✅ 新增：记录模型配置
        wandb.config.update({
            "hidden_dim": 256,
            "fusion_dim": TEXT_DIM + ACOUSTIC_DIM + VISUAL_DIM,
            "num_lines": 12,
            "bert_frozen": True,
        })
    except Exception as e:
        print(f"WandB初始化失败: {e}")

    try:
        data, _ = load_data(args.dataset)
        expected_folds = [f'fold{i}' for i in range(1, 6)]
        for fold_key in expected_folds:
            if fold_key not in data:
                raise ValueError(f"数据集缺少 {fold_key}")
        print(f"✅ 数据集完整性检查通过，包含所有5折数据")
    except Exception as e:
        print(f"❌ 数据集预检查失败: {e}")
        return

    all_folds_results = []
    
    # --- 5折交叉验证循环 ---
    for fold_id in range(1, 6):
        print(f"\n{'#'*60}")
        print(f"### 开始训练 FOLD {fold_id} ###")
        print(f"{'#'*60}\n")
        
        try:
            # 设置数据加载器
            train_dataloader, valid_dataloader, num_train_optimization_steps = set_up_data_loader(fold_id)
            
            # 准备模型
            model = prep_for_training(num_train_optimization_steps)
            
            # 训练模型
            fold_results = train(
                model=model,
                train_dataloader=train_dataloader,
                validation_dataloader=valid_dataloader,
                num_train_optimization_steps=num_train_optimization_steps,
                fold_id=fold_id,
                run_timestamp=run_timestamp
            )
            
            fold_results["fold_id"] = fold_id
            all_folds_results.append(fold_results)
            
            # ✅ 修复：添加三分类指标到打印输出
            print(f"\n{'#'*60}")
            print(f"### FOLD {fold_id} 训练完成 ###")
            print(f"  最佳MAE: {fold_results['best_mae']:.4f}")
            print(f"  最佳二分类准确率: {fold_results['best_acc']:.4f}")
            print(f"  最佳五分类准确率: {fold_results['best_acc_7']:.4f}")
            print(f"  最佳三分类准确率: {fold_results['best_triclass_accuracy']:.4f}")  # 新增
            print(f"  最佳F1分数: {fold_results['best_f_score']:.4f}")
            print(f"  最佳相关系数: {fold_results['best_corr']:.4f}")
            print(f"  最佳五分类F1: {fold_results['best_multiclass_f1']:.4f}")
            print(f"  最佳三分类F1: {fold_results['best_triclass_f1']:.4f}")         # 新增
            print(f"  最佳epoch: {fold_results['best_epoch']}")                        # 新增
            print(f"{'#'*60}\n")
            
        except Exception as e:
            print(f"\n❌ FOLD {fold_id} 训练过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            all_folds_results.append({
                "fold_id": fold_id,
                "error": str(e)
            })

    print(f"\n{'#'*60}")
    print(f"### 5折交叉验证汇总 ###")
    print(f"{'#'*60}\n")
    
    successful_folds = [r for r in all_folds_results if "error" not in r]
    
    if successful_folds:
        # ✅ 修复：添加三分类指标到平均结果
        avg_results = {
            "avg_mae": np.mean([r["best_mae"] for r in successful_folds]),
            "avg_acc": np.mean([r["best_acc"] for r in successful_folds]),
            "avg_acc_7": np.mean([r["best_acc_7"] for r in successful_folds]),
            "avg_triclass_accuracy": np.mean([r["best_triclass_accuracy"] for r in successful_folds]),  # 新增
            "avg_f_score": np.mean([r["best_f_score"] for r in successful_folds]),
            "avg_corr": np.mean([r["best_corr"] for r in successful_folds]),
            "avg_multiclass_f1": np.mean([r["best_multiclass_f1"] for r in successful_folds]),
            "avg_triclass_f1": np.mean([r["best_triclass_f1"] for r in successful_folds]),            # 新增
        }
        
        print(f"平均MAE: {avg_results['avg_mae']:.4f}")
        print(f"平均二分类准确率: {avg_results['avg_acc']:.4f}")
        print(f"平均五分类准确率: {avg_results['avg_acc_7']:.4f}")
        print(f"平均三分类准确率: {avg_results['avg_triclass_accuracy']:.4f}")  # 新增
        print(f"平均F1分数: {avg_results['avg_f_score']:.4f}")
        print(f"平均相关系数: {avg_results['avg_corr']:.4f}")
        print(f"平均五分类F1: {avg_results['avg_multiclass_f1']:.4f}")
        print(f"平均三分类F1: {avg_results['avg_triclass_f1']:.4f}")            # 新增
    else:
        print("❌ 所有折的训练都失败了，无法计算平均结果")
        avg_results = "N/A"

    # 将最终的平均结果保存到文件
    final_summary = {
        "args": vars(args),
        "run_timestamp": run_timestamp,
        "individual_fold_results": all_folds_results,
        "average_results": avg_results if successful_folds else "N/A"
    }
    
    os.makedirs("run_artifacts", exist_ok=True)
    final_summary_path = os.path.join("run_artifacts", f"run_{run_timestamp}_CV_summary.json")
    with open(final_summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 交叉验证最终摘要已保存到: {final_summary_path}")
    
if __name__ == "__main__":
    main()