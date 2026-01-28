import torch
from torch import nn
from torch.nn import L1Loss
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
import math
import logging

try:
    from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM
except ImportError:
    TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = 768, 74, 35

logger = logging.getLogger(__name__)

# ==========================================
# 1. Basic Components (Positional Encoding, Transformer, VIB, MIL)
# ==========================================

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

class VariationalIB(nn.Module):
    """
    [Section 3.3] Unimodal Feature Purification via VIB
    """
    def __init__(self, input_dim, z_dim=256):
        super(VariationalIB, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc_mu = nn.Linear(input_dim, z_dim)
        self.fc_logvar = nn.Linear(input_dim, z_dim)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        
        z = self.reparameterize(mu, log_var)
        
        # KL Divergence: D_KL(N(mu, sigma) || N(0, I))
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss.sum(dim=-1).mean() 
        
        return z, kl_loss

class GatedAttentionMIL(nn.Module):
    """
    [Section 3.5] Attention-based MIL Aggregation
    """
    def __init__(self, dim, reduction=16, dropout=0.1):
        super(GatedAttentionMIL, self).__init__()
        self.linear_V = nn.Linear(dim, dim // reduction)
        self.linear_U = nn.Linear(dim, dim // reduction)
        self.attention_weights = nn.Linear(dim // reduction, 1)
        
        self.norm = nn.LayerNorm(dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Seq_Len/Instances, Dim)
        # Eq (5): Attention Weights calculation
        A_V = torch.tanh(self.linear_V(x))
        A_U = torch.sigmoid(self.linear_U(x))
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1)
        
        # Aggregation
        M = torch.sum(A * x, dim=1)
        
        M = self.norm(M)
        if M.shape[0] > 1:
             M = self.bn(M)
        M = self.dropout(M)
        
        return M, A

class ModalityGating(nn.Module):
    """
    [Section 3.4] Global-Local Cross-Modal Gating
    """
    def __init__(self, dim, context_dim, hidden_dim=64, dropout=0.1):
        super(ModalityGating, self).__init__()
        self.local_proj = nn.Linear(dim, hidden_dim)
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
        batch_size, seq_len, _ = x.shape
        local_h = self.local_proj(x)
        # Expand global context to match sequence length
        global_h = self.global_proj(global_context).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Eq (3): Compute Gate
        gate = self.gate_generator(local_h + global_h)
        
        # Eq (4): Apply Gate
        gated_x = x * gate
        residual_out = self.residual(x)
        return self.norm(gated_x + residual_out), gate

class EnhancedAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.3, ff_dim=512):
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
        seq = torch.stack([text_feat, acoustic_feat, visual_feat], dim=1)
        attn_out, _ = self.attn(seq, seq, seq)
        fused = attn_out.mean(dim=1) 
        concat = torch.cat([text_feat, acoustic_feat, visual_feat], dim=-1)
        cross_feat = self.cross_modal(concat)
        residual = self.residual(fused)
        return self.norm(fused + cross_feat + residual)

# ==========================================
# 2. DIB-MIL
# ==========================================

class DIB(BertPreTrainedModel):
    def __init__(self, config, multimodal_config=None, loss_function=None):
        super(DIB, self).__init__(config)
        self.config = config
        self.vib_lambda = getattr(multimodal_config, 'vib_lambda', 1e-4)
        self.bert = BertModel(config)
        self.loss_function = loss_function if loss_function is not None else nn.L1Loss()
        
        PROJ_DIM = 256 
        
        # === 1. Feature Extraction (Sec 3.2) ===
        # Text: BERT -> Linear Projection
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_DIM, PROJ_DIM),
            nn.LayerNorm(PROJ_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Audio/Visual: 1D-CNNs
        # Input (Batch, Dim, Seq) -> Conv1d -> (Batch, Proj, Seq)
        self.acoustic_cnn = nn.Conv1d(ACOUSTIC_DIM, PROJ_DIM, kernel_size=3, padding=1)
        self.visual_cnn = nn.Conv1d(VISUAL_DIM, PROJ_DIM, kernel_size=3, padding=1)
        
        # === 2. Transformer Encoders (Sec 3.2) ===
        # Capture temporal dependencies
        self.pos_encoding = PositionalEncoding(PROJ_DIM)
        self.text_transformer = TransformerBlock(PROJ_DIM, num_heads=4, ff_dim=512)
        self.acoustic_transformer = TransformerBlock(PROJ_DIM, num_heads=4, ff_dim=512)
        self.visual_transformer = TransformerBlock(PROJ_DIM, num_heads=4, ff_dim=512)
        
        # Removed LSTM layers as they are not in the paper's specific methodology
        ENC_DIM = PROJ_DIM 
        
        # === 3. VIB Purification (Sec 3.3) ===
        self.vib_text = VariationalIB(ENC_DIM, z_dim=ENC_DIM)
        self.vib_audio = VariationalIB(ENC_DIM, z_dim=ENC_DIM)
        self.vib_visual = VariationalIB(ENC_DIM, z_dim=ENC_DIM)

        # === 4. Global-Local Gating (Sec 3.4) ===
        CONTEXT_DIM = ENC_DIM * 3
        self.modality_gate_text = ModalityGating(ENC_DIM, CONTEXT_DIM)
        self.modality_gate_audio = ModalityGating(ENC_DIM, CONTEXT_DIM)
        self.modality_gate_visual = ModalityGating(ENC_DIM, CONTEXT_DIM)
        
        # === 5. Fusion ===
        self.cross_fusion = EnhancedAttentionFusion(embed_dim=ENC_DIM, num_heads=4)
        
        # === 6. MIL Aggregation (Sec 3.5) ===
        MIL_INPUT_DIM = ENC_DIM * 3 + ENC_DIM 
        self.mil_aggregator = GatedAttentionMIL(dim=MIL_INPUT_DIM, reduction=8)
        
        # === 7. Regression Head ===
        self.regressor = nn.Sequential(
            nn.Linear(MIL_INPUT_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, token_type_ids=None, label_ids=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        
        # --- 1. Feature Extraction ---
        # Text
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        text_feat = self.text_proj(outputs.last_hidden_state) # (B, S, Dim)
        
        # Audio & Visual: Expand -> Permute -> 1D-CNN -> Permute Back
        if acoustic.dim() == 2: acoustic = acoustic.unsqueeze(1).expand(-1, seq_len, -1)
        if visual.dim() == 2: visual = visual.unsqueeze(1).expand(-1, seq_len, -1)
        
        # (B, S, D) -> (B, D, S) for Conv1d
        acoustic_feat = self.acoustic_cnn(acoustic.permute(0, 2, 1)).permute(0, 2, 1)
        visual_feat = self.visual_cnn(visual.permute(0, 2, 1)).permute(0, 2, 1)
        
        # --- 2. Temporal Modeling (Transformer) ---
        text_seq = self.text_transformer(self.pos_encoding(text_feat.transpose(0,1)).transpose(0,1))
        acoustic_seq = self.acoustic_transformer(self.pos_encoding(acoustic_feat.transpose(0,1)).transpose(0,1))
        visual_seq = self.visual_transformer(self.pos_encoding(visual_feat.transpose(0,1)).transpose(0,1))
        
        # --- 3. VIB Purification ---
        # Encoder is implicitly handled by Transformer+MLP inside VIB
        z_text, kl_t = self.vib_text(text_seq)
        z_audio, kl_a = self.vib_audio(acoustic_seq)
        z_visual, kl_v = self.vib_visual(visual_seq)
        
        total_kl_loss = kl_t + kl_a + kl_v
        
        # --- 4. Global Context & Gating ---
        # Global Context: Average of purified features
        global_text = z_text.mean(dim=1)
        global_audio = z_audio.mean(dim=1)
        global_visual = z_visual.mean(dim=1)
        global_context = torch.cat([global_text, global_audio, global_visual], dim=-1)
        
        text_gated, _ = self.modality_gate_text(z_text, global_context)
        audio_gated, _ = self.modality_gate_audio(z_audio, global_context)
        visual_gated, _ = self.modality_gate_visual(z_visual, global_context)
        
        # --- 5. Fusion & MIL Aggregation ---
        fused_global = self.cross_fusion(global_text, global_audio, global_visual)
        fused_seq = fused_global.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Instance Construction: Concatenate features
        instance_features = torch.cat([text_gated, audio_gated, visual_gated, fused_seq], dim=-1)
        
        # MIL Pooling -> Bag Representation
        bag_embedding, attention_scores = self.mil_aggregator(instance_features)
        
        # --- 6. Prediction ---
        logits = self.regressor(bag_embedding)

        if label_ids is not None:
            # Sec 3.6: Optimization Objective
            loss_task = self.loss_function(logits.view(-1), label_ids.view(-1))
            return loss_task + self.vib_lambda * total_kl_loss
        
        return logits, attention_scores

    def test(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
