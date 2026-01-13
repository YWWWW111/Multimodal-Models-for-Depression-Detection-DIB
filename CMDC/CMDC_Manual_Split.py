import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import random # 导入random模块

# ================= 配置参数 =================
data_root = "E:/Depression/CMDC"
parts = {
    'part1': ('HC01', 'HC10'),
    'part2': ('HC11', 'HC20'),
    'part3': ('HC21', 'HC30'),
    'part4': ('HC31', 'HC40'),
    'part5': ('HC41', 'HC52'),
    'part6': ('MDD01', 'MDD10'),
    'part7': ('MDD11', 'MDD20'),
    'part8': ('MDD21', 'MDD26')
}

# ================= 全局异常日志收集器 =================
anomaly_logs = {
    "missing_files": [],
    "dimension_mismatch": [],
    "empty_modality": []
}
# 新增：用于跟踪部分数据缺失的全局字典
partial_data_tracker = {}

# ================= 工具函数 =================
def parse_range(range_str):
    """解析范围字符串如 1-20 或 13-26 & 1-6"""
    if not isinstance(range_str, str):
        return []
        
    ids = []
    for part in range_str.split('&'):
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                ids.extend([f"{i:02d}" for i in range(start, end+1)])
            except ValueError:
                continue
        else:
            try:
                ids.append(f"{int(part):02d}")
            except ValueError:
                continue
    return ids

def create_phq_dict():
    """创建PHQ分数字典"""
    phq_scores = {
        'HC01': 2, 'HC02': 1, 'HC03': 2, 'HC04': 1, 'HC05': 2,
        'HC06': 2, 'HC07': 2, 'HC08': 1, 'HC09': 5, 'HC10': 0,
        'HC11': 2, 'HC12': 0, 'HC13': 5, 'HC14': 3, 'HC15': 2,
        'HC16': 5, 'HC17': 3, 'HC18': 0, 'HC19': 2, 'HC20': 3,
        'HC21': 0, 'HC22': 0, 'HC23': 2, 'HC24': 3, 'HC25': 5,
        'HC26': 1, 'HC27': 2, 'HC28': 3, 'HC29': 0, 'HC30': 0,
        'HC31': 0, 'HC32': 0, 'HC33': 0, 'HC34': 4, 'HC35': 1,
        'HC36': 0, 'HC37': 1, 'HC38': 1, 'HC39': 0, 'HC40': 0,
        'HC41': 0, 'HC42': 0, 'HC43': 2, 'HC44': 0, 'HC45': 0,
        'HC46': 0, 'HC47': 1, 'HC48': 2, 'HC49': 2, 'HC50': 1,
        'HC51': 4, 'HC52': 1,
        'MDD01': 13, 'MDD02': 21, 'MDD03': 12, 'MDD04': 20, 'MDD05': 21,
        'MDD06': 18, 'MDD07': 18, 'MDD08': 13, 'MDD09': 25, 'MDD10': 18,
        'MDD11': 12, 'MDD12': 20, 'MDD13': 9, 'MDD14': 23, 'MDD15': 11,
        'MDD16': 11, 'MDD17': 17, 'MDD18': 13, 'MDD19': 23, 'MDD20': 19,
        'MDD21': 23, 'MDD22': 16, 'MDD23': 12, 'MDD24': 17, 'MDD25': 12, 'MDD26': 18
    }
    return phq_scores

# ================= 加载元数据 =================
phq_scores = create_phq_dict()
print(f"成功加载 {len(phq_scores)} 个被试的PHQ分数")

# ================= 手动定义5折验证集 =================
manual_folds = [
    {
        'HC': ['HC10', 'HC21', 'HC31', 'HC36', 'HC02', 'HC26', 'HC01', 'HC11', 'HC43', 'HC14', 'HC09'],
        'MDD': ['MDD16', 'MDD01', 'MDD07', 'MDD04', 'MDD21']
    },
    {
        'HC': ['HC12', 'HC22', 'HC32', 'HC39', 'HC04', 'HC35', 'HC03', 'HC15', 'HC19', 'HC17', 'HC13'],
        'MDD': ['MDD15', 'MDD08', 'MDD10', 'MDD12', 'MDD19']
    },
    {
        'HC': ['HC18', 'HC29', 'HC33', 'HC08', 'HC37', 'HC05', 'HC06', 'HC48', 'HC20', 'HC16'],
        'MDD': ['MDD03', 'MDD18', 'MDD22', 'MDD02', 'MDD14']
    },
    {
        'HC': ['HC30', 'HC40', 'HC41', 'HC42', 'HC38', 'HC47', 'HC07', 'HC49', 'HC24', 'HC25'],
        'MDD': ['MDD13', 'MDD11', 'MDD06', 'MDD17', 'MDD05', 'MDD20']
    },
    {
        'HC': ['HC44', 'HC45', 'HC46', 'HC50', 'HC52', 'HC23', 'HC27', 'HC28', 'HC34', 'HC51'],
        'MDD': ['MDD23', 'MDD25', 'MDD24', 'MDD26', 'MDD09']
    }
]

# ================= 根据手动验证集生成训练集 =================
# 1. 获取所有被试ID
all_subject_ids = list(phq_scores.keys())
print(f"\n总共找到 {len(all_subject_ids)} 名被试。")

# 2. 构建Fold数据结构
folds = []
for i, val_fold in enumerate(manual_folds):
    # 验证集：手动指定的ID
    val_mdd = val_fold['MDD']
    val_hc = val_fold['HC']
    val_ids = val_mdd + val_hc
    
    # 训练集：所有被试 - 验证集
    train_ids = [sid for sid in all_subject_ids if sid not in val_ids]
    train_mdd = [sid for sid in train_ids if 'MDD' in sid]
    train_hc = [sid for sid in train_ids if 'HC' in sid]
    
    folds.append({
        'name': f'Fold {i+1}',
        'train': {'MDD': train_mdd, 'HC': train_hc},
        'val': {'MDD': val_mdd, 'HC': val_hc}
    })

# 打印5折划分信息
print("\n已生成手动定义的5折交叉验证划分:")
for fold in folds:
    print(f"- {fold['name']}: "
          f"训练集({len(fold['train']['MDD']) + len(fold['train']['HC'])}个) = "
          f"MDD({len(fold['train']['MDD'])}), HC({len(fold['train']['HC'])}); "
          f"验证集({len(fold['val']['MDD']) + len(fold['val']['HC'])}个) = "
          f"MDD({len(fold['val']['MDD'])}), HC({len(fold['val']['HC'])})")


# 在主处理流程之前添加多模态数据可用性检查
def check_modality_available(subject_path, modality='all'):
    """
    检查受试者是否有可用的数据
    参数:
        subject_path: 被试数据路径
        modality: 'vision', 'audio', 'text', 或 'all'
    返回:
        如果modality='all', 返回包含所有模态可用性的字典
        否则返回指定模态的可用性布尔值
    """
    availability = {
        'vision': False,
        'audio': False,
        'text': False
    }
    
    for q in range(1, 13):
        # 检查视觉数据
        vision_path = os.path.join(subject_path, f'Q{q}.npy')
        if os.path.exists(vision_path):
            availability['vision'] = True
            
        # 检查音频数据
        audio_path = os.path.join(subject_path, f'Q{q}.pkl')
        if os.path.exists(audio_path):
            availability['audio'] = True
            
        # 检查文本数据
        text_path = os.path.join(subject_path, f'Q{q}.txt')
        if os.path.exists(text_path):
            availability['text'] = True
        
        # 如果所有模态都找到了，可以提前退出循环
        if all(availability.values()):
            break
    
    return availability if modality == 'all' else availability.get(modality, False)

# 特征维度定义
feature_dims = {
    'audio': 128,   # VGGish特征维度
    'vision': 768   # Timesformer特征维度
}

# ================= 检查特征维度 =================
def check_feature_dimensions(sample_path):
    """检查一个样本的所有模态特征维度"""
    print("\n=== 特征维度检查 ===")
    
    # 视觉特征
    vision_file = os.path.join(sample_path, "Q1.npy")
    if os.path.exists(vision_file):
        try:
            vision_feat = np.load(vision_file)
            print(f"\n视觉特征 (Q1.npy):")
            print(f"- 形状: {vision_feat.shape}")
            print(f"- 类型: {vision_feat.dtype}")
            print(f"- 文件大小: {os.path.getsize(vision_file) / 1024:.2f} KB")
        except Exception as e:
            print(f"读取视觉特征出错: {str(e)}")
            
    # 音频特征
    audio_file = os.path.join(sample_path, "Q1.pkl")
    if os.path.exists(audio_file):
        try:
            with open(audio_file, 'rb') as f:
                audio_feat = pickle.load(f)
            if isinstance(audio_feat, np.ndarray):
                print(f"\n音频特征 (Q1.pkl):")
                print(f"- 形状: {audio_feat.shape}")
                print(f"- 类型: {audio_feat.dtype}")
                print(f"- 文件大小: {os.path.getsize(audio_file) / 1024:.2f} KB")
        except Exception as e:
            print(f"读取音频特征出错: {str(e)}")
            
    # 文本特征
    text_file = os.path.join(sample_path, "Q1.txt")
    if os.path.exists(text_file):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_feat = f.read()
            print(f"\n文本特征 (Q1.txt):")
            print(f"- 内容预览: {text_feat[:100]}...")
            print(f"- 文件大小: {os.path.getsize(text_file)} bytes")
        except Exception as e:
            print(f"读取文本特征出错: {str(e)}")

# 在主处理流程之前检查特征维度
print("\n检查特征维度...")
for part in parts:
    part_path = os.path.join(data_root, part)
    if os.path.exists(part_path):
        for subject_dir in os.listdir(part_path):
            subject_path = os.path.join(part_path, subject_dir)
            if os.path.isdir(subject_path) and any([os.path.exists(os.path.join(subject_path, f)) for f in ["Q1.npy", "Q1.pkl", "Q1.txt"]]):
                print(f"\n检查被试 {subject_dir} 的特征维度:")
                check_feature_dimensions(subject_path)
                break  # 只需要检查一个有数据的样本即可
        break

# ================= 加载特征数据 =================
def load_features(subject_path, subject_id):
    """加载单个被试的多模态特征，并记录异常"""
    # 修改：预初始化为包含12个None的列表，用于占位
    features = {
        'text': [None] * 12, 
        'audio': [None] * 12, 
        'vision': [None] * 12
    }
    
    # 为当前被试初始化数据存在性跟踪器
    presence = {
        'vision': [False] * 12,
        'audio': [False] * 12,
        'text': [False] * 12
    }

    for q in range(1, 13):
        q_index = q - 1 # 列表索引为 0-11

        # 1. 加载视觉特征 (使用Timesformer特征)
        vision_path = os.path.join(subject_path, f'Q{q}.npy')
        if os.path.exists(vision_path):
            try:
                vision_feat = np.load(vision_path)
                if isinstance(vision_feat, np.ndarray):
                    feat_mean = np.nanmean(vision_feat, axis=0) if len(vision_feat.shape) > 1 else vision_feat
                        
                    if len(feat_mean) != feature_dims['vision']:
                        msg = f"Subject {subject_id}, Vision, Q{q}: 维度不匹配, 期望 {feature_dims['vision']}, 实际 {len(feat_mean)}"
                        print(f"警告: {msg}")
                        anomaly_logs["dimension_mismatch"].append(msg)
                        # 维度不对也填零
                        features['vision'][q_index] = np.zeros(feature_dims['vision'], dtype=np.float32)
                    else:
                        features['vision'][q_index] = np.nan_to_num(feat_mean, 0.0)
                        presence['vision'][q_index] = True # 记录数据存在
            except Exception as e:
                print(f"警告: 读取视觉特征文件 {vision_path} 时出错: {str(e)}")
                features['vision'][q_index] = np.zeros(feature_dims['vision'], dtype=np.float32)
        else:
            # 记录缺失文件
            msg = f"Subject {subject_id}, Vision, Q{q}: 文件缺失，已自动填充零向量 (Shape: {feature_dims['vision']})"
            print(f"  [INFO] {msg}") # <--- 实时打印填充信息
            anomaly_logs["missing_files"].append(msg)
            # 缺失文件填零
            features['vision'][q_index] = np.zeros(feature_dims['vision'], dtype=np.float32)
        
        # 2. 加载音频特征 (使用VGGish特征)
        audio_path = os.path.join(subject_path, f'Q{q}.pkl')
        if os.path.exists(audio_path):
            try:
                with open(audio_path, 'rb') as f:
                    audio_feat = pickle.load(f)
                if isinstance(audio_feat, np.ndarray):
                    # 修复之前的语法错误
                    feat_mean = np.nanmean(audio_feat, axis=0) if len(audio_feat.shape) > 1 else audio_feat
                        
                    if len(feat_mean) != feature_dims['audio']:
                        msg = f"Subject {subject_id}, Audio, Q{q}: 维度不匹配, 期望 {feature_dims['audio']}, 实际 {len(feat_mean)}"
                        print(f"警告: {msg}")
                        anomaly_logs["dimension_mismatch"].append(msg)
                        features['audio'][q_index] = np.zeros(feature_dims['audio'], dtype=np.float32)
                    else:
                        features['audio'][q_index] = np.nan_to_num(feat_mean, 0.0)
                        presence['audio'][q_index] = True # 记录数据存在
                else:
                    print(f"警告: 音频特征格式不正确 {audio_path}")
                    features['audio'][q_index] = np.zeros(feature_dims['audio'], dtype=np.float32)
            except Exception as e:
                print(f"警告: 读取音频特征文件 {audio_path} 时出错: {str(e)}")
                features['audio'][q_index] = np.zeros(feature_dims['audio'], dtype=np.float32)
        else:
            # 记录缺失文件
            msg = f"Subject {subject_id}, Audio, Q{q}: 文件缺失，已自动填充零向量 (Shape: {feature_dims['audio']})"
            print(f"  [INFO] {msg}") # <--- 实时打印填充信息
            anomaly_logs["missing_files"].append(msg)
            features['audio'][q_index] = np.zeros(feature_dims['audio'], dtype=np.float32)

        # 3. 加载文本特征 (原始文本)
        text_path = os.path.join(subject_path, f'Q{q}.txt')
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                    if text_content:
                        features['text'][q_index] = text_content
                        presence['text'][q_index] = True # 记录数据存在
                    else:
                        features['text'][q_index] = "" # 空内容
            except Exception as e:
                print(f"警告: 读取文本文件 {text_path} 时出错: {str(e)}")
                features['text'][q_index] = ""
        else:
            # 记录缺失文件
            msg = f"Subject {subject_id}, Text, Q{q}: 文件缺失，已自动填充空字符串"
            # print(f"  [INFO] {msg}") # 文本缺失比较常见，可以不打印以免刷屏，或者取消注释
            anomaly_logs["missing_files"].append(msg)
            features['text'][q_index] = ""
    
    # 将当前被试的存在性记录存入全局跟踪器
    partial_data_tracker[subject_id] = presence

    # 处理特征
    processed_features = {}
    # 处理视觉和音频特征
    for mod in ['audio', 'vision']:
        # 此时 features[mod] 已经是一个长度为12的列表，包含了数据或零向量
        # 直接转换为 numpy 数组，形状必然是 (12, dim)
        stacked = np.array(features[mod])
        processed_features[mod] = np.asarray(stacked, dtype=np.float32)
        
        # 检查是否全为零（即整个模态缺失）
        if not any(presence[mod]):
             msg = f"Subject {subject_id}: {mod} 模态无任何有效特征, 全部为零填充"
             anomaly_logs["empty_modality"].append(msg)

    # 处理文本特征
    # 修改：不再拼接为字符串，而是保留为长度为12的列表，以保持与Audio/Vision的(12, dim)对齐
    processed_features['text'] = features['text'] 
    
    # 检查文本是否全为空
    if not any(features['text']):
        msg = f"Subject {subject_id}: text 模态无任何有效特征, 全部为空字符串"
        anomaly_logs["empty_modality"].append(msg)
        
    return processed_features

# 添加文件检查函数
def check_feature_files(subject_path):
    """检查特征文件的可用性"""
    available_features = {
        'vision': False,
        'audio': False,
        'text': False
    }
    
    for q in range(1, 13):
        vision_path = os.path.join(subject_path, f'Q{q}.csv')
        audio_path = os.path.join(subject_path, f'Q{q}_audio.csv')
        text_path = os.path.join(subject_path, f'Q{q}_text.csv')
        
        if os.path.exists(vision_path):
            available_features['vision'] = True
        if os.path.exists(audio_path):
            available_features['audio'] = True
        if os.path.exists(text_path):
            available_features['text'] = True
    
    return available_features

# 在主处理流程中添加
for part in parts:
    part_path = os.path.join(data_root, part)
    if os.path.exists(part_path):
        print(f"\n检查目录 {part_path} 的特征文件:")
        for subject_dir in os.listdir(part_path):
            subject_path = os.path.join(part_path, subject_dir)
            if os.path.isdir(subject_path):
                available = check_feature_files(subject_path)
                print(f"\n{subject_dir}:")
                for mod, exists in available.items():
                    print(f"- {mod}: {'可用' if exists else '不可用'}")
        break

# ================= 主处理流程 =================
all_data = {}
for fold in folds:
    fold_name = fold['name'].replace(' ', '').lower()
    all_data[fold_name] = {'train': [], 'val': []}  # 改为 train 和 val
    
    print(f"\n{'='*20} 处理 {fold['name']} {'='*20}")
    
    # 1. 先加载该折的所有数据
    for split in ['train', 'val']:  # 改为 train 和 val
        print(f"\n--- 处理 {split} 集 ---")
        
        subject_ids_to_process = fold[split]['MDD'] + fold[split]['HC']
        
        for subject_id in subject_ids_to_process:
            # 找到被试所在的part目录
            found = False
            for part_name, (start_id, end_id) in parts.items():
                # 提取ID中的数字部分
                prefix = ''.join(filter(str.isalpha, subject_id))
                num = int(''.join(filter(str.isdigit, subject_id)))
                start_prefix = ''.join(filter(str.isalpha, start_id))
                start_num = int(''.join(filter(str.isdigit, start_id)))
                end_prefix = ''.join(filter(str.isalpha, end_id))
                end_num = int(''.join(filter(str.isdigit, end_id)))

                if prefix == start_prefix and start_num <= num <= end_num:
                    subject_path = os.path.join(data_root, part_name, subject_id)
                    
                    print(f"处理被试 {subject_id} (来自 {part_name})")
                    # 传入 subject_id 用于日志记录
                    features = load_features(subject_path, subject_id)
                    
                    # 创建样本记录，并包含ID
                    sample_record = {
                        'id': subject_id, # 关键：保存被试ID
                        'text': features['text'], 
                        'audio': features['audio'].astype(np.float32),
                        'vision': features['vision'].astype(np.float32),
                        'label': float(phq_scores[subject_id])
                    }
                    
                    all_data[fold_name][split].append(sample_record)
                    found = True
                    break
            
            if not found:
                print(f"警告: 未能在任何part中找到被试 {subject_id} 的目录")

    # 2. 新增：在该折内进行标准化 (Standardization)
    # 注意：必须只在训练集上 fit，然后 transform 训练集和验证集，以防止数据泄露
    print(f"\n正在对 {fold_name} 进行特征标准化...")
    
    for modality in ['audio', 'vision']:
        # 收集所有训练数据的特征用于 fit
        # 形状转换: [样本数, 12, dim] -> [样本数*12, dim]
        train_features = np.concatenate([s[modality] for s in all_data[fold_name]['train']], axis=0)
        
        # 初始化并拟合 Scaler
        scaler = StandardScaler()
        scaler.fit(train_features)
        
        print(f"  - {modality} 训练集均值: {scaler.mean_[:5]}... (前5维)")
        print(f"  - {modality} 训练集方差: {scaler.var_[:5]}... (前5维)")
        
        # 变换训练集
        for sample in all_data[fold_name]['train']:
            # transform 输入 (12, dim) -> 输出 (12, dim)
            sample[modality] = scaler.transform(sample[modality])
            
        # 变换验证集
        for sample in all_data[fold_name]['val']:  # 改为 val
            sample[modality] = scaler.transform(sample[modality])

# 添加数据统计信息
print("\n数据统计信息:")
for fold_name, fold_content in all_data.items():
    print(f"\n{fold_name}:")
    for split, samples in fold_content.items():
        num_samples = len(samples)
        print(f"  - {split} 集样本数量: {num_samples}")
        if num_samples > 0:
            labels = [sample['label'] for sample in samples]
            print(f"    标签范围: [{min(labels):.2f}, {max(labels):.2f}]")

# 确保保存目录存在再写入
import os
os.makedirs('datasets', exist_ok=True)
# 使用新文件名以避免覆盖旧文件
with open('datasets/CMDC_Text_CV_SL.pkl', 'wb') as f:
    pickle.dump(all_data, f)

print("\n预处理完成！数据已按标准5折交叉验证格式保存为 datasets/CMDC_Text_CV_SL.pkl")

# ================= 最终异常摘要 =================
def print_anomaly_summary():
    """打印所有收集到的异常信息摘要"""
    print("\n\n" + "="*80)
    print(" " * 30 + "数据预处理异常摘要")
    print("="*80)

    # 1. 缺失文件
    if anomaly_logs["missing_files"]:
        print(f"\n--- 缺失的特征文件 ({len(anomaly_logs['missing_files'])} 条) ---")

        # 新增：计算并打印各模态缺失文件矩阵
        missing_counts = {'text': 0, 'audio': 0, 'vision': 0}
        for log in anomaly_logs["missing_files"]:
            if ", Vision," in log:
                missing_counts['vision'] += 1
            elif ", Audio," in log:
                missing_counts['audio'] += 1
            elif ", Text," in log:
                missing_counts['text'] += 1
        
        print("\n[缺失文件统计矩阵]")
        print("-" * 40)
        print(f"| {'模态':<10} | {'缺失文件数':<20} |")
        print("-" * 40)
        print(f"| {'Vision':<10} | {missing_counts['vision']:<20} |")
        print(f"| {'Audio':<10} | {missing_counts['audio']:<20} |")
        print(f"| {'Text':<10} | {missing_counts['text']:<20} |")
        print("-" * 40)

        # 为了简洁，只打印前20条
        print("\n[缺失文件详情 (部分)]")
        for log in anomaly_logs["missing_files"][:2500]:
            print(log)
        if len(anomaly_logs["missing_files"]) > 2500:
            print(f"... (还有 {len(anomaly_logs['missing_files']) - 2500} 条未显示)")
    else:
        print("\n✅ [检查通过] 未发现任何缺失的特征文件。")

    # 2. 维度不匹配
    if anomaly_logs["dimension_mismatch"]:
        print(f"\n--- 特征维度不匹配 ({len(anomaly_logs['dimension_mismatch'])} 条) ---")
        for log in anomaly_logs["dimension_mismatch"]:
            print(log)
    else:
        print("\n✅ [检查通过] 所有特征文件维度均正确。")

    # 3. 模态为空（生成了零向量）
    if anomaly_logs["empty_modality"]:
        print(f"\n--- 因无有效特征而生成零向量的模态 ({len(anomaly_logs['empty_modality'])} 条) ---")
        for log in anomaly_logs["empty_modality"]:
            print(log)
    else:
        print("\n✅ [检查通过] 所有被试的每个模态都至少有一个有效的特征文件。")
    
    # 新增：分析并打印部分数据缺失的情况
    print(f"\n--- 部分模态数据缺失总结 ---")
    partial_data_found = False
    for subject_id, modalities in sorted(partial_data_tracker.items()):
        subject_has_partial_data = False
        output_lines = []
        for mod, presence_list in modalities.items():
            num_present = sum(presence_list)
            # 核心逻辑：数据存在，但又不是全部12个都存在
            if 0 < num_present < 12:
                if not subject_has_partial_data:
                    subject_has_partial_data = True
                    partial_data_found = True
                
                missing_qs = [str(i + 1) for i, present in enumerate(presence_list) if not present]
                # 添加填充说明
                fill_note = "已填充零向量" if mod != 'text' else "已填充空字符串"
                output_lines.append(f"  - {mod.capitalize():<7}: 存在 {num_present}/12 个, 缺失问题: [{', '.join(missing_qs)}] -> {fill_note}")
        
        if subject_has_partial_data:
            print(f"被试: {subject_id}")
            for line in output_lines:
                print(line)

    if not partial_data_found:
        print("✅ [检查通过] 未发现任何模态存在部分数据缺失的情况 (即所有存在的模态数据都是完整的12个问题)。")

    print("\n" + "="*80)
    print("摘要结束")
    print("="*80)

# 在脚本最后调用异常摘要函数
print_anomaly_summary()

# ================= 数据量统计分析 =================
def print_data_volume_analysis():
    """详细分析三个模态的数据量，用于诊断过拟合问题"""
    print("\n\n" + "="*80)
    print(" " * 25 + "多模态数据量统计分析")
    print("="*80)
    
    # 1. 全局统计
    total_subjects = len(partial_data_tracker)
    total_possible_samples = total_subjects * 12  # 每个被试12个问题
    
    print(f"\n【全局数据规模】")
    print(f"  总被试数量: {total_subjects}")
    print(f"  理论最大样本数 (被试数 × 12问题): {total_possible_samples}")
    
    # 2. 各模态实际数据量统计
    modality_stats = {
        'vision': {'total': 0, 'missing': 0, 'partial_subjects': 0, 'complete_subjects': 0},
        'audio': {'total': 0, 'missing': 0, 'partial_subjects': 0, 'complete_subjects': 0},
        'text': {'total': 0, 'missing': 0, 'partial_subjects': 0, 'complete_subjects': 0}
    }
    
    for subject_id, modalities in partial_data_tracker.items():
        for mod, presence_list in modalities.items():
            num_present = sum(presence_list)
            num_missing = 12 - num_present;
            
            modality_stats[mod]['total'] += num_present
            modality_stats[mod]['missing'] += num_missing
            
            if num_present == 12:
                modality_stats[mod]['complete_subjects'] += 1
            elif num_present > 0:
                modality_stats[mod]['partial_subjects'] += 1
    
    print(f"\n【各模态数据完整性统计】")
    print("-" * 80)
    print(f"{'模态':<10} | {'有效样本数':<12} | {'缺失样本数':<12} | {'数据完整率':<12} | {'完整被试':<10} | {'部分被试':<10}")
    print("-" * 80)
    
    for mod in ['vision', 'audio', 'text']:
        stats = modality_stats[mod]
        total = stats['total']
        missing = stats['missing']
        completeness = (total / total_possible_samples) * 100 if total_possible_samples > 0 else 0
        
        print(f"{mod.capitalize():<10} | {total:<12} | {missing:<12} | {completeness:<11.2f}% | "
              f"{stats['complete_subjects']:<10} | {stats['partial_subjects']:<10}")
    
    # 3. 计算多模态对齐的有效样本
    aligned_samples = 0
    for subject_id, modalities in partial_data_tracker.items():
        for i in range(12):
            if all(modalities[mod][i] for mod in ['vision', 'audio', 'text']):
                aligned_samples += 1
    
    print(f"\n【多模态对齐情况】")
    print(f"  三模态完全对齐的样本数: {aligned_samples} / {total_possible_samples} ({aligned_samples/total_possible_samples*100:.2f}%)")
    print(f"  说明: 只有这些样本的三个模态数据都存在，可以进行完整的多模态融合")
    
    # 4. 按HC和MDD分组统计
    hc_stats = {'vision': 0, 'audio': 0, 'text': 0, 'count': 0}
    mdd_stats = {'vision': 0, 'audio': 0, 'text': 0, 'count': 0}
    
    for subject_id, modalities in partial_data_tracker.items():
        if 'HC' in subject_id:
            hc_stats['count'] += 1
            for mod in ['vision', 'audio', 'text']:
                hc_stats[mod] += sum(modalities[mod])
        else:
            mdd_stats['count'] += 1
            for mod in ['vision', 'audio', 'text']:
                mdd_stats[mod] += sum(modalities[mod])
    
    print(f"\n【按类别分组统计】")
    print("-" * 60)
    print(f"{'类别':<10} | {'被试数':<8} | {'Vision':<10} | {'Audio':<10} | {'Text':<10}")
    print("-" * 60)
    
    for label, stats in [('HC (健康)', hc_stats), ('MDD (抑郁)', mdd_stats)]:
        print(f"{label:<10} | {stats['count']:<8} | {stats['vision']:<10} | "
              f"{stats['audio']:<10} | {stats['text']:<10}")
    
    # 5. 数据规模 vs 模型参数建议
    print(f"\n【过拟合风险评估】")
    total_effective_samples = min(modality_stats['vision']['total'], 
                                   modality_stats['audio']['total'], 
                                   modality_stats['text']['total'])
    
    print(f"  最小模态有效样本数: {total_effective_samples}")
    print(f"  5折交叉验证训练集规模: 约 {int(total_effective_samples * 0.8)} 样本/折")
    print(f"  5折交叉验证验证集规模: 约 {int(total_effective_samples * 0.2)} 样本/折")
    
    print(f"\n  ⚠️  过拟合风险分析:")
    if total_effective_samples < 500:
        print(f"     - 数据量较小 (<500)，建议:")
        print(f"       1. 使用较小的模型 (例如: 隐藏层维度 ≤ 256)")
        print(f"       2. 增加 Dropout 比例 (0.3-0.5)")
        print(f"       3. 使用强正则化 (L2 weight decay = 1e-3 ~ 1e-2)")
        print(f"       4. 减少训练轮数，使用早停 (Early Stopping)")
        print(f"       5. 考虑数据增强策略")
    elif total_effective_samples < 1000:
        print(f"     - 数据量中等 (500-1000)，建议:")
        print(f"       1. 使用中等规模模型 (隐藏层维度 256-512)")
        print(f"       2. 适度 Dropout (0.2-0.3)")
        print(f"       3. 使用 L2 正则化 (weight decay = 1e-4)")
    else:
        print(f"     - 数据量充足 (>1000)，可以使用更大模型")
    
    # 6. 每折数据量统计
    print(f"\n【5折交叉验证数据分布】")
    print("-" * 60)
    print(f"{'Fold':<8} | {'训练集':<15} | {'验证集':<15} | {'比例':<10}")
    print("-" * 60)
    
    for fold in folds:
        train_count = len(fold['train']['MDD']) + len(fold['train']['HC'])
        val_count = len(fold['val']['MDD']) + len(fold['val']['HC'])
        ratio = f"{train_count}/{val_count}"
        
        print(f"{fold['name']:<8} | {train_count:<15} | {val_count:<15} | {ratio:<10}")
    
    print("\n" + "="*80)
    print("数据量分析结束")
    print("="*80)

# 在 print_anomaly_summary() 之后调用
print_data_volume_analysis()

# ================= 数据分布诊断函数 =================
def diagnose_fold_distribution():
    """诊断每折的类别分布和标签分布"""
    print("\n\n" + "="*80)
    print(" " * 25 + "5折数据分布诊断报告")
    print("="*80)
    
    print("\n【类别分布检查】")
    print("-" * 80)
    print(f"{'Fold':<8} | {'训练集HC':<10} | {'训练集MDD':<12} | {'验证集HC':<10} | {'验证集MDD':<12} | {'验证集MDD比例':<15}")
    print("-" * 80)
    
    fold_imbalance_scores = []
    
    for fold in folds:
        train_hc = len(fold['train']['HC'])
        train_mdd = len(fold['train']['MDD'])
        val_hc = len(fold['val']['HC'])
        val_mdd = len(fold['val']['MDD'])
        
        # 计算验证集中MDD的比例
        val_total = val_hc + val_mdd
        val_mdd_ratio = (val_mdd / val_total * 100) if val_total > 0 else 0
        
        # 不平衡度评分：理想情况下MDD应该占33.3% (26/78)
        imbalance_score = abs(val_mdd_ratio - 33.33)
        fold_imbalance_scores.append(imbalance_score)
        
        print(f"{fold['name']:<8} | {train_hc:<10} | {train_mdd:<12} | {val_hc:<10} | "
              f"{val_mdd:<12} | {val_mdd_ratio:<14.2f}%")
    
    avg_imbalance = np.mean(fold_imbalance_scores)
    max_imbalance = np.max(fold_imbalance_scores)
    
    print(f"\n  平均不平衡度: {avg_imbalance:.2f}% (理想值: 0%)")
    print(f"  最大不平衡度: {max_imbalance:.2f}%")
    
    if max_imbalance > 10:
        print(f"  ⚠️  警告: 存在严重的类别不平衡 (>10%)，建议使用分层采样!")
    
    # 标签分布检查
    print(f"\n【PHQ标签分布检查】")
    print("-" * 80)
    print(f"{'Fold':<8} | {'训练集均值':<12} | {'训练集方差':<12} | {'验证集均值':<12} | {'验证集方差':<12}")
    print("-" * 80)
    
    for fold in folds:
        train_ids = fold['train']['MDD'] + fold['train']['HC']
        val_ids = fold['val']['MDD'] + fold['val']['HC']
        
        train_labels = [phq_scores[sid] for sid in train_ids]
        val_labels = [phq_scores[sid] for sid in val_ids]
        
        train_mean = np.mean(train_labels)
        train_var = np.var(train_labels)
        val_mean = np.mean(val_labels)
        val_var = np.var(val_labels)
        
        print(f"{fold['name']:<8} | {train_mean:<12.2f} | {train_var:<12.2f} | "
              f"{val_mean:<12.2f} | {val_var:<12.2f}")
    
    print("\n" + "="*80)
    print("诊断报告结束")
    print("="*80)

# 在 print_data_volume_analysis() 之后调用
diagnose_fold_distribution()

# ================= 分析保存的数据 =================
def analyze_folds_data(data_path):
    """分析处理后的多折数据集结构和大小"""
    print("\n=== 开始分析多折数据集 ===")
    
    # 用于全局统计所有Fold的数据
    # 修改：存储 (length, id) 元组
    global_text_stats = {i: [] for i in range(12)}

    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        for fold_name, fold_content in data.items():
            print(f"\n{fold_name}:")
            for split, samples in fold_content.items():
                print(f"\n  {split} 集样本数: {len(samples)}")
                if not samples:
                    continue
                # 关键：确保每个样本都包含ID
                ids = [s.get('id', 'N/A') for s in samples]
                print(f"    部分ID: {ids[:5]}...")

                # --- 文本长度详细统计 ---
                # 检查第一个样本的 text 类型
                sample_text = samples[0]['text']
                if isinstance(sample_text, list):
                    print(f"  [文本长度统计 (字符数)]")
                    print(f"    {'Q_ID':<5} | {'Mean':<8} | {'Max':<5} | {'Min':<5} | {'Empty Count'}")
                    print("    " + "-"*55)
                    
                    # 收集当前split的统计数据
                    current_split_stats = {i: [] for i in range(12)}
                    for s in samples:
                        texts = s['text']
                        for i in range(12):
                            l = len(texts[i])
                            current_split_stats[i].append(l)
                            # 修改：同时记录长度和ID，用于全局统计
                            global_text_stats[i].append((l, s['id'])) 
                    
                    # 打印当前split的统计表
                    for i in range(12):
                        lens = current_split_stats[i]
                        if lens:
                            mean_l = np.mean(lens)
                            max_l = np.max(lens)
                            min_l = np.min(lens)
                            empty_c = sum(1 for l in lens if l == 0)
                            print(f"    Q{i+1:<4} | {mean_l:<8.2f} | {max_l:<5} | {min_l:<5} | {empty_c}")
                else:
                    # 兼容旧格式
                    text_lengths = [len(s['text']) for s in samples if s['text']]
                    if text_lengths:
                        print(f"- text  : 字符串, 平均长度={np.mean(text_lengths):.2f}")
                    else:
                        print("- text  : 无有效文本数据")

                # --- 修改开始: 适应 Sequence Level 的分析 ---
                # 由于现在每个样本的形状可能不同 (N不同)，不能直接 np.stack
                # 我们打印形状的统计信息
                audio_shapes = [s['audio'].shape for s in samples]
                vision_shapes = [s['vision'].shape for s in samples]
                
                # 计算平均序列长度
                avg_audio_seq_len = np.mean([s[0] for s in audio_shapes])
                avg_vision_seq_len = np.mean([s[0] for s in vision_shapes])

                print(f"- audio : 序列长度均值={avg_audio_seq_len:.2f}, 特征维度={audio_shapes[0][1]}")
                print(f"- vision: 序列长度均值={avg_vision_seq_len:.2f}, 特征维度={vision_shapes[0][1]}")
                
                # 简单的数值统计 (展平后计算)
                all_audio_vals = np.concatenate([s['audio'].flatten() for s in samples])
                all_vision_vals = np.concatenate([s['vision'].flatten() for s in samples])
                
                print(f"  Audio数值统计: 均值={all_audio_vals.mean():.4f}, 方差={all_audio_vals.var():.4f}")
                print(f"  Vision数值统计: 均值={all_vision_vals.mean():.4f}, 方差={all_vision_vals.var():.4f}")
                # --- 修改结束 ---

                labels = np.array([s['label'] for s in samples])
                print(f"- labels: 范围 [{labels.min():.2f}, {labels.max():.2f}], 均值={labels.mean():.2f}, 方差={labels.var():.2f}")

        # --- 打印全局文本长度建议 ---
        print("\n" + "="*65)
        print("全局文本长度统计 (所有Folds汇总)")
        print("="*65)
        # 修改表头，增加 Max ID
        print(f"{'Q_ID':<5} | {'Mean':<8} | {'Max':<5} | {'Max ID':<10} | {'95%ile':<6}")
        print("-" * 60)
        all_lens = []
        for i in range(12):
            stats = global_text_stats[i] # 这是一个包含 (len, id) 的列表
            if stats:
                # 分离长度列表用于计算统计量
                lens = [x[0] for x in stats]
                all_lens.extend(lens)
                
                mean_l = np.mean(lens)
                
                # 找到最大值及其对应的ID
                max_val, max_id = max(stats, key=lambda x: x[0])
                
                p95 = np.percentile(lens, 95)
                print(f"Q{i+1:<4} | {mean_l:<8.2f} | {max_val:<5} | {max_id:<10} | {int(p95):<6}")
        
        if all_lens:
            print("-" * 60)
            print(f"所有问题汇总: Max={max(all_lens)}, Mean={np.mean(all_lens):.2f}, 95%ile={int(np.percentile(all_lens, 95))}")
            print(f"建议设置 max_sequence_length 至少为: {int(np.percentile(all_lens, 95)) + 10} (覆盖95%样本) 或 {max(all_lens)} (覆盖所有)")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {str(e)}")
        import traceback
        traceback.print_exc()

# 添加特征验证函数
def validate_features(features):
    """验证特征数据是否有效"""
    for mod, feat in features.items():
        if np.all(feat == 0):
            print(f"警告: {mod} 特征全为零")
        else:
            print(f"{mod} 特征统计:")
            print(f"- 最小值: {np.min(feat):.4f}")
            print(f"- 最大值: {np.max(feat):.4f}")
            print(f"- 均值: {np.mean(feat):.4f}")
            print(f"- 非零元素比例: {np.mean(feat != 0)*100:.2f}%")

# 分析新保存的数据
print("\n开始分析新保存的数据...")
analyze_folds_data('datasets/CMDC_Text_CV_SL.pkl')