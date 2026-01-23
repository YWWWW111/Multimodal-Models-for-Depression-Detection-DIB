import torch
from torch import nn
from torch.nn import L1Loss
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class GatedAttentionMIL(nn.Module):
    """
    MIL Core: 解决关键病理特征稀疏问题
    Attention机制自动为具有抑郁特征的片段分配高权重
    """
    def __init__(self, dim, reduction=16, num_heads=4, dropout=0.1):
        super(GatedAttentionMIL, self).__init__()
        self.linear_V = nn.Linear(dim, dim // reduction)
        self.linear_U = nn.Linear(dim, dim // reduction)
        self.attention_weights = nn.Linear(dim // reduction, 1)
        
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        # 1. Gated Attention Pooling (Ilse et al., 2018)
        A_V = torch.tanh(self.linear_V(x))
        A_U = torch.sigmoid(self.linear_U(x))
        A = self.attention_weights(A_V * A_U) # (Batch, Seq, 1)
        A = torch.softmax(A, dim=1)           # 归一化权重，找出最关键的 instances
        
        # 2. Self-Attention 增强实例间的特征交互
        attn_out, _ = self.multihead_attn(x, x, x)
        x_refined = x + attn_out
        
        # 3. 加权聚合 (Bag Representation)
        M = torch.sum(A * x_refined, dim=1)   # (Batch, Dim)
        
        M = self.norm(M)
        if M.shape[0] > 1: # BN需要batch > 1
             M = self.bn(M)
        M = self.dropout(M)
        
        return M, A

class ModalityGating(nn.Module):
    """
    修正版: 真正的跨模态感知门控
    解决 'Smiling Depression' 和 噪声模态问题
    """
    def __init__(self, dim, context_dim, hidden_dim=32, dropout=0.1):
        super(ModalityGating, self).__init__()
        # 局部特征变换
        self.local_proj = nn.Linear(dim, hidden_dim)
        # 全局上下文变换 (来自所有模态的融合信息)
        self.global_proj = nn.Linear(context_dim, hidden_dim)
        
        self.gate_generator = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )
        self.residual = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, global_context):
        """
        x: (Batch, Seq, Dim) - 局部时序特征
        global_context: (Batch, Context_Dim) - 全局多模态上下文
        """
        # 计算门控权重: 基于局部特征 + 全局一致性
        # global_context 需要扩展到序列长度
        batch_size, seq_len, _ = x.shape
        
        local_h = self.local_proj(x) # (B, S, H)
        global_h = self.global_proj(global_context).unsqueeze(1).expand(-1, seq_len, -1) # (B, S, H)
        
        # 融合局部与全局信息生成权重
        # 如果局部特征与全局上下文冲突（例如视觉在笑，但全局是悲伤），权重会降低
        weights = self.gate_generator(local_h + global_h)
        
        gated_x = x * weights
        residual_out = self.residual(x)
        output = self.norm(gated_x + residual_out)
        
        return output, weights

class EnhancedAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.3, ff_dim=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_modal = nn.Sequential(
            nn.Linear(embed_dim * 3, ff_dim),
            nn.LayerNorm(ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.residual = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_feat, acoustic_feat, visual_feat):
        # text_feat: (B, Dim) - 这里输入的是全局均值
        # 堆叠成 (Batch, 3, Dim) 序列
        seq = torch.stack([text_feat, acoustic_feat, visual_feat], dim=1)
        
        # 利用 Self-Attention 学习模态间的相关性
        attn_out, _ = self.attn(seq, seq, seq)
        fused = attn_out.mean(dim=1) # (B, Dim)
        
        concat = torch.cat([text_feat, acoustic_feat, visual_feat], dim=-1)
        cross_feat = self.cross_modal(concat)
        residual = self.residual(fused)
        return self.norm(fused + cross_feat + residual)

def create_projection_layers(input_dim, output_dim, num_layers=3, dropout=0.2):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(input_dim if i == 0 else output_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class DIB(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, loss_function=None):
        super(DIB, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.loss_function = loss_function if loss_function is not None else nn.L1Loss()
        
        # === 1. 特征投影 (各模态 -> 512维) ===
        PROJ_DIM = 512
        self.text_proj = create_projection_layers(TEXT_DIM, PROJ_DIM, num_layers=3)
        self.acoustic_proj = create_projection_layers(ACOUSTIC_DIM, PROJ_DIM, num_layers=3)
        self.visual_proj = create_projection_layers(VISUAL_DIM, PROJ_DIM, num_layers=3)
        
        # === 2. 序列增强 (Transformer) ===
        self.pos_encoding = PositionalEncoding(PROJ_DIM)
        self.text_transformer = TransformerBlock(PROJ_DIM, num_heads=4, ff_dim=1024)
        self.acoustic_transformer = TransformerBlock(PROJ_DIM, num_heads=4, ff_dim=1024)
        self.visual_transformer = TransformerBlock(PROJ_DIM, num_heads=4, ff_dim=1024)
        
        # === 3. 实例编码 (Bi-LSTM) ===
        LSTM_HIDDEN = 256
        self.text_encoder = nn.LSTM(PROJ_DIM, LSTM_HIDDEN, batch_first=True, bidirectional=True, num_layers=1)
        self.acoustic_encoder = nn.LSTM(PROJ_DIM, LSTM_HIDDEN, batch_first=True, bidirectional=True, num_layers=1)
        self.visual_encoder = nn.LSTM(PROJ_DIM, LSTM_HIDDEN, batch_first=True, bidirectional=True, num_layers=1)
        # LSTM 输出维度 = 256 * 2 = 512
        ENC_DIM = LSTM_HIDDEN * 2
        
        # === 4. 模态门控 (应对噪声和Smiling Depression) ===
        # 上下文维度 = 3个模态的全局均值拼在一起 = 512 * 3 = 1536
        CONTEXT_DIM = ENC_DIM * 3
        self.modality_gate_text = ModalityGating(ENC_DIM, CONTEXT_DIM)
        self.modality_gate_audio = ModalityGating(ENC_DIM, CONTEXT_DIM)
        self.modality_gate_visual = ModalityGating(ENC_DIM, CONTEXT_DIM)
        
        # === 5. 跨模态全局融合 ===
        self.cross_fusion = EnhancedAttentionFusion(embed_dim=ENC_DIM, num_heads=4)
        
        # === 6. MIL聚合 (解决稀疏特征) ===
        # 输入维度: Text(512) + Audio(512) + Visual(512) + Fused(512) = 2048
        MIL_INPUT_DIM = ENC_DIM * 3 + ENC_DIM 
        self.mil_aggregator = GatedAttentionMIL(dim=MIL_INPUT_DIM, reduction=8)
        
        # === 7. 回归器 ===
        self.regressor = nn.Sequential(D
            nn.Linear(MIL_INPUT_DIM, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, token_type_ids=None, label_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        
        # 1. BERT & Projection
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        text_seq = self.text_proj(outputs.last_hidden_state)
        
        # Expand modalities to match sequence length if necessary
        if acoustic.dim() == 2: acoustic = acoustic.unsqueeze(1).expand(-1, seq_len, -1)
        if visual.dim() == 2: visual = visual.unsqueeze(1).expand(-1, seq_len, -1)
        
        acoustic_seq = self.acoustic_proj(acoustic)
        visual_seq = self.visual_proj(visual)
        
        # 2. Transformer Encoding
        text_seq = self.text_transformer(self.pos_encoding(text_seq.transpose(0,1)).transpose(0,1))
        acoustic_seq = self.acoustic_transformer(self.pos_encoding(acoustic_seq.transpose(0,1)).transpose(0,1))
        visual_seq = self.visual_transformer(self.pos_encoding(visual_seq.transpose(0,1)).transpose(0,1))
        
        # 3. Instance Encoding (LSTM)
        text_enc, _ = self.text_encoder(text_seq)       # (batch, seq, 512)
        acoustic_enc, _ = self.acoustic_encoder(acoustic_seq)
        visual_enc, _ = self.visual_encoder(visual_seq)
        
        # 4. Construct Global Context for Gating
        # 计算每个模态的全局表示 (Mean Pooling)
        global_text = text_enc.mean(dim=1)
        global_audio = acoustic_enc.mean(dim=1)
        global_visual = visual_enc.mean(dim=1)
        # 全局上下文: 融合了三者信息
        global_context = torch.cat([global_text, global_audio, global_visual], dim=-1) # (Batch, 1536)
        
        # 5. Modality Gating (利用全局上下文校准局部帧)
        text_gated, w_t = self.modality_gate_text(text_enc, global_context)
        audio_gated, w_a = self.modality_gate_audio(acoustic_enc, global_context)
        visual_gated, w_v = self.modality_gate_visual(visual_enc, global_context)
        
        # 6. Global Cross-Modal Fusion
        fused_global = self.cross_fusion(global_text, global_audio, global_visual) # (Batch, 512)
        fused_seq = fused_global.unsqueeze(1).expand(-1, seq_len, -1) # 扩展回序列
        
        # 7. Concatenate Instances
        # (Batch, Seq, 512*4) = (Batch, Seq, 2048)
        instance_features = torch.cat([text_gated, audio_gated, visual_gated, fused_seq], dim=-1)
        
        # 8. MIL Aggregation (Bag-level prediction)
        # 自动寻找最具区分度的时间片，忽略无关片段
        bag_embedding, attention_scores = self.mil_aggregator(instance_features)
        
        # 9. Regression
        logits = self.regressor(bag_embedding)

        if label_ids is not None:
            return self.loss_function(logits.view(-1), label_ids.view(-1))
        
        return logits, attention_scores

    def test(self, *args, **kwargs):
        return self.forward(*args, **kwargs)