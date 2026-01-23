import torch
from torch import nn
from torch.nn import L1Loss
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from modules.transformer import TransformerEncoder
from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.sparse.linalg
import logging

logger = logging.getLogger(__name__)

# 通用投影维度
COMMON_DIM = 50

class bottleneckFusion(nn.Module):
    """基于注意力的瓶颈融合模块"""
    def __init__(self, num_latents, in_size, output_dim, hidden=50, dropout=0.5):
        super(bottleneckFusion, self).__init__()
        self.dropout = nn.Dropout(dropout) 
        self.num_latents = num_latents
        # 潜在变量：作为跨模态信息交换的枢纽
        self.latents = nn.Parameter(torch.empty(1, num_latents, in_size).normal_(std=0.02))  
        self.scale_l = nn.Parameter(torch.zeros(1))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))

    def attention(self, q, k, v): 
        """标准的缩放点积注意力"""
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) 
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x

    def fusion_btn(self, l1, a1, v1):
        """单次瓶颈融合迭代"""
        BS = l1.shape[0]
        
        # 处理输入维度：确保是 (Batch, Seq, Dim)
        if l1.dim() == 2:
            l1 = l1.unsqueeze(1)
            a1 = a1.unsqueeze(1)
            v1 = v1.unsqueeze(1)
            
        # 拼接所有模态用于Cross Attention
        concat_avt = torch.cat((l1, a1, v1), dim=1)
        
        # 扩展 latents 到当前 Batch 大小
        latents = self.latents.to(l1.device).expand(BS, -1, -1)

        # Step 1: Latents 从多模态特征中提取共享信息
        fused_avt_latents = self.attention(q=latents, k=concat_avt, v=concat_avt)

        # Step 2: 各模态反向查询 Latents，注入共享信息
        l1 = l1 + self.scale_l * self.attention(q=l1, k=fused_avt_latents, v=fused_avt_latents)  
        a1 = a1 + self.scale_a * self.attention(q=a1, k=fused_avt_latents, v=fused_avt_latents)
        v1 = v1 + self.scale_v * self.attention(q=v1, k=fused_avt_latents, v=fused_avt_latents)
        
        return l1, a1, v1

    def forward(self, l1, a1, v1):
        """迭代 3 次进行深度融合"""
        for i in range(3):
            l1, a1, v1 = self.fusion_btn(l1, a1, v1)
            
        # 提取融合后的文本表示（作为主要输出）
        if l1.dim() == 3:
            y_l = l1[:, -1, :] if l1.size(1) > 1 else l1.squeeze(1)
        else:
            y_l = l1
        return y_l


class VIBFusion(nn.Module):
    """变分信息瓶颈融合模块：通过互信息正则化实现鲁棒的多模态融合"""
    def __init__(self, num_latents, dim):
        super().__init__()
        self.d_l = dim
        self.num_latents = num_latents
        
        # VIB Encoders: 将各模态特征映射到隐变量的参数空间
        self.encoder_l = nn.Sequential(nn.Linear(self.d_l, 1024), nn.ReLU(inplace=True))
        self.encoder_a = nn.Sequential(nn.Linear(self.d_l, 1024), nn.ReLU(inplace=True))
        self.encoder_v = nn.Sequential(nn.Linear(self.d_l, 1024), nn.ReLU(inplace=True))
        self.encoder_f = nn.Sequential(nn.Linear(self.d_l, 1024), nn.ReLU(inplace=True))

        # 均值和标准差预测层
        self.fc_mu_l, self.fc_std_l = nn.Linear(1024, self.d_l), nn.Linear(1024, self.d_l)
        self.fc_mu_a, self.fc_std_a = nn.Linear(1024, self.d_l), nn.Linear(1024, self.d_l)
        self.fc_mu_v, self.fc_std_v = nn.Linear(1024, self.d_l), nn.Linear(1024, self.d_l)
        self.fc_mu_f, self.fc_std_f = nn.Linear(1024, self.d_l), nn.Linear(1024, self.d_l)
        
        # Decoder: 从隐变量预测最终输出
        self.decoder = nn.Linear(self.d_l, 1)

        # 瓶颈融合模块
        self.fusion1 = bottleneckFusion(self.num_latents, self.d_l, self.d_l) 

    def reparameterise(self, mu, std):
        """重参数化技巧：使采样过程可微分"""
        eps = torch.randn_like(std)
        return mu + std * eps

    # ==================== 互信息估计辅助函数 ====================
    def pairwise_distances(self, x):
        """计算批次内样本的成对距离"""
        bn = x.shape[0]
        x = x.view(bn, -1)
        instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    def calculate_gram_mat(self, x, sigma):
        """计算高斯核矩阵"""
        dist = self.pairwise_distances(x)
        return torch.exp(-dist / sigma)

    def calculate_lowrank(self, A, alpha, k, v):
        """基于低秩近似计算 Rényi 熵"""
        n = A.shape[0]
        if v.shape[0] != n:
            v = np.random.randn(n)
             
        try:
            A_np = A.detach().cpu().numpy()
            k_eff = min(k, n - 1)
            if k_eff < 1:
                k_eff = 1
            # 使用稀疏特征分解
            _, U = scipy.sparse.linalg.eigsh(A_np, k=k_eff, v0=v, ncv=min(n, k_eff * 2 + 1), tol=1e-1)
            U = torch.from_numpy(U).to(A.device).float()
            # 在 PyTorch 中计算特征值以保留梯度
            eigs = torch.clamp(torch.linalg.eigvalsh(torch.mm(U.t(), torch.mm(A, U))), min=0)
        except Exception as e:
            logger.warning(f"Low-rank approximation failed: {e}")
            return torch.zeros(1, device=A.device)
            
        tr = torch.sum(eigs ** alpha) + (n - k) * torch.clamp((1 - torch.sum(eigs)) / (n - k), min=0) ** alpha
        return (1 / (1 - alpha)) * torch.log2(tr + 1e-8)

    def calculate_MI_lowrank(self, x, y, s_x, s_y, alpha, k, v):
        """计算互信息的低秩近似"""
        ky = self.calculate_gram_mat(y, s_y)
        ky = ky / (torch.trace(ky) + 1e-6)
        return self.calculate_lowrank(ky, alpha, k, v)

    def forward(self, x_l, x_a, x_v):
        """前向传播：VIB 编码 -> 融合 -> 解码"""
        bs = x_l.shape[0]
        
        # 1. 各模态的 VIB 编码
        feat_l = self.encoder_l(x_l)
        mu_l, std_l = self.fc_mu_l(feat_l), F.softplus(self.fc_std_l(feat_l) - 5, beta=1)
        z_l = self.reparameterise(mu_l, std_l)
        
        feat_a = self.encoder_a(x_a)
        mu_a, std_a = self.fc_mu_a(feat_a), F.softplus(self.fc_std_a(feat_a) - 5, beta=1)
        z_a = self.reparameterise(mu_a, std_a)
        
        feat_v = self.encoder_v(x_v)
        mu_v, std_v = self.fc_mu_v(feat_v), F.softplus(self.fc_std_v(feat_v) - 5, beta=1)
        z_v = self.reparameterise(mu_v, std_v)
        
        # 2. 动态计算核宽度 sigma（基于数据分布）
        def get_sigma(z):
            try:
                Z_numpy = z.cpu().detach().numpy().reshape(bs, -1)
                k_z = squareform(pdist(Z_numpy, 'euclidean'))
                k_neighbors = min(6, bs)
                return np.mean(np.mean(np.sort(k_z, axis=1)[:, 1:k_neighbors]))
            except:
                return 1.0

        sigma_z_l = get_sigma(z_l)
        sigma_z_a = get_sigma(z_a)
        sigma_z_v = get_sigma(z_v)

        # 3. 计算各模态的互信息正则化损失
        query = np.random.randn(bs)
        k_mi = min(10, bs - 1)
        
        loss_l = 1e-5 * self.calculate_MI_lowrank(x_l, z_l, 1, sigma_z_l ** 2, 1.9, k_mi, query)
        loss_a = 1e-5 * self.calculate_MI_lowrank(x_a, z_a, 1, sigma_z_a ** 2, 1.9, k_mi, query)
        loss_v = 1e-5 * self.calculate_MI_lowrank(x_v, z_v, 65, sigma_z_v ** 2, 1.9, k_mi, query)

        # 4. 瓶颈融合
        outputf = self.fusion1(z_l, z_a, z_v)
        
        # 5. 融合后的 VIB 编码
        f_feat = self.encoder_f(outputf)
        mu, std = self.fc_mu_f(f_feat), F.softplus(self.fc_std_f(f_feat) - 5, beta=1)
        z = self.reparameterise(mu, std)
        
        # 6. 最终预测
        output = self.decoder(z)
        
        # 7. 计算融合表示的互信息损失
        sigma_z = get_sigma(z)
        loss_f = 1e-5 * self.calculate_MI_lowrank(outputf, z, 1000, sigma_z ** 2, 1.9, k_mi, query)
        
        # 总的 VIB 正则化损失
        total_vib_loss = loss_l + loss_a + loss_v + loss_f
        
        return output, total_vib_loss

    def test(self, x_l, x_a, x_v):
        """测试模式：直接使用均值，不采样"""
        feat_l = self.encoder_l(x_l)
        mu_l, std_l = self.fc_mu_l(feat_l), F.softplus(self.fc_std_l(feat_l) - 5, beta=1)
        z_l = mu_l  # 测试时使用均值

        feat_a = self.encoder_a(x_a)
        mu_a, std_a = self.fc_mu_a(feat_a), F.softplus(self.fc_std_a(feat_a) - 5, beta=1)
        z_a = mu_a

        feat_v = self.encoder_v(x_v)
        mu_v, std_v = self.fc_mu_v(feat_v), F.softplus(self.fc_std_v(feat_v) - 5, beta=1)
        z_v = mu_v
 
        outputf = self.fusion1(z_l, z_a, z_v)

        f_feat = self.encoder_f(outputf)
        mu = self.fc_mu_f(f_feat)
        output = self.decoder(mu)
    
        return output


class DIB(BertPreTrainedModel):
    """深度信息瓶颈多模态融合模型"""
    def __init__(self, config, multimodal_config=None):
        super().__init__(config)
        self.d_l = COMMON_DIM
        self.num_lines = 12
        
        # 1. 文本编码器 (BERT)
        self.bert = BertModel(config)
        # 文本投影: BERT hidden size (768) -> COMMON_DIM (50)
        self.proj_l = nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. 音频/视觉 投影与时序编码
        self.proj_a = nn.Conv1d(ACOUSTIC_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_v = nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        
        # TransformerEncoder 处理 12 行的时序依赖
        self.transa = TransformerEncoder(embed_dim=self.d_l, num_heads=5, layers=3, 
                                         attn_dropout=0.5, relu_dropout=0.3, 
                                         res_dropout=0.3, embed_dropout=0.2, attn_mask=False)
        self.transv = TransformerEncoder(embed_dim=self.d_l, num_heads=5, layers=3,
                                         attn_dropout=0.5, relu_dropout=0.3,
                                         res_dropout=0.3, embed_dropout=0.2, attn_mask=False)
        
        # 3. 信息瓶颈融合模块
        self.fusion = VIBFusion(num_latents=4, dim=self.d_l)
        
        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, 
                token_type_ids=None, line_mask=None, label_ids=None, **kwargs):
        """
        input_ids: (Batch, 12, SeqLen) - 12行文本的token IDs
        visual: (Batch, 12, VisDim) - 12行的视觉特征
        acoustic: (Batch, 12, AcouDim) - 12行的声学特征
        line_mask: (Batch, 12) - 标记有效行
        label_ids: (Batch,) - PHQ-9 分数标签
        """
        batch_size, num_lines, seq_len = input_ids.shape
        
        # === 1. 文本处理 ===
        # 展平 (Batch*12, SeqLen) 以输入 BERT
        input_ids_flat = input_ids.view(-1, seq_len)
        mask_flat = attention_mask.view(-1, seq_len) if attention_mask is not None else None
        token_flat = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        
        bert_output = self.bert(input_ids_flat, attention_mask=mask_flat, token_type_ids=token_flat)
        # 序列输出: (Batch*12, SeqLen, 768)
        text_seq = bert_output.last_hidden_state
        
        # 投影: Conv1d 需要 (N, C, L)
        text_emb = text_seq.permute(0, 2, 1)  # -> (Batch*12, 768, SeqLen)
        text_emb = self.proj_l(text_emb)      # -> (Batch*12, 50, SeqLen)
        
        # 池化：取最后一个时间步
        x_l = text_emb[:, :, -1].view(batch_size, num_lines, -1)  # -> (Batch, 12, 50)
        
        # === 2. 音频处理 ===
        # (Batch, 12, DimA) -> (Batch, DimA, 12)
        x_a_in = acoustic.permute(0, 2, 1) 
        x_a = self.proj_a(x_a_in)          # -> (Batch, 50, 12)
        x_a = x_a.permute(2, 0, 1)         # -> (12, Batch, 50) for Transformer
        
        # === 3. 视觉处理 ===
        x_v_in = visual.permute(0, 2, 1)
        x_v = self.proj_v(x_v_in)          # -> (Batch, 50, 12)
        x_v = x_v.permute(2, 0, 1)         # -> (12, Batch, 50)
        
        # === 4. 时序建模 ===
        out_a = self.transa(x_a)           # (12, Batch, 50)
        out_v = self.transv(x_v)           # (12, Batch, 50)
        
        # 取最后一个时间步作为整体表示
        final_a = out_a[-1]                # (Batch, 50)
        final_v = out_v[-1]
        final_l = x_l[:, -1, :]            # 取第12行
        
        # === 5. 应用 Line Mask（可选：增强鲁棒性）===
        if line_mask is not None:
            mask = line_mask[:, -1].unsqueeze(-1).to(final_l.device)
            final_l = final_l * mask
            # 注意：音频/视觉已经过 Transformer 整合，不需单独 mask
        
        # === 6. VIB 融合与预测 ===
        pred, loss_vib = self.fusion(final_l, final_a, final_v)
        
        # === 7. 损失计算 ===
        if label_ids is not None:
            # 回归任务损失 (L1 Loss)
            loss_task = L1Loss()(pred.view(-1), label_ids.view(-1))
            # 总损失 = 任务损失 + VIB 正则化损失
            return loss_task + loss_vib
             
        return pred

    def test(self, input_ids, visual, acoustic, attention_mask=None, 
             token_type_ids=None, line_mask=None, **kwargs):
        """测试模式：不计算损失，直接返回预测"""
        batch_size, num_lines, seq_len = input_ids.shape
        
        # 复用 forward 的特征提取逻辑
        input_ids_flat = input_ids.view(-1, seq_len)
        mask_flat = attention_mask.view(-1, seq_len) if attention_mask is not None else None
        token_flat = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        
        bert_output = self.bert(input_ids_flat, attention_mask=mask_flat, token_type_ids=token_flat)
        text_seq = bert_output.last_hidden_state
        
        text_emb = text_seq.permute(0, 2, 1)
        text_emb = self.proj_l(text_emb)
        x_l = text_emb[:, :, -1].view(batch_size, num_lines, -1)
        
        x_a_in = acoustic.permute(0, 2, 1) 
        x_a = self.proj_a(x_a_in)
        x_a = x_a.permute(2, 0, 1)
        
        x_v_in = visual.permute(0, 2, 1)
        x_v = self.proj_v(x_v_in)
        x_v = x_v.permute(2, 0, 1)
        
        out_a = self.transa(x_a)
        out_v = self.transv(x_v)
        
        final_a = out_a[-1]
        final_v = out_v[-1]
        final_l = x_l[:, -1, :]
        
        # 测试模式下使用 VIBFusion.test()
        return self.fusion.test(final_l, final_a, final_v)
