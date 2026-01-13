import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from global_configs import TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM

class DIB(BertPreTrainedModel):
    def __init__(self, config, multimodal_config=None):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # 固定为12行文本
        self.num_lines = 12
        
        # 融合后的特征维度: Text + Audio + Vision
        # 768 + 128 + 768 = 1664
        self.fusion_dim = TEXT_DIM + ACOUSTIC_DIM + VISUAL_DIM
        
        # 最简单的回归器: Flatten所有行的特征 -> 线性层 -> 分数
        # 输入维度: 12 * 1664 = 19968
        # self.classifier = nn.Linear(self.num_lines * self.fusion_dim, 1)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.fc1 = nn.Linear(self.num_lines * self.fusion_dim, 256)
        self.fc2 = nn.Linear(256, 1)

        self.init_weights()

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        line_mask=None,
        label_ids=None,
        **kwargs
    ):
        """
        input_ids: (batch_size, 12, seq_len)
        visual: (batch_size, 12, visual_dim)
        acoustic: (batch_size, 12, acoustic_dim)
        line_mask: (batch_size, 12) - 标记哪些行是真实的，哪些是padding
        """
        batch_size, num_lines, seq_len = input_ids.shape
        
        # 1. 处理文本 (Text)
        # 将 (Batch, 12, Seq_Len) 展平为 (Batch*12, Seq_Len) 以输入 BERT
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len) if attention_mask is not None else None
        token_type_ids_flat = token_type_ids.view(-1, seq_len) if token_type_ids is not None else None
        
        outputs = self.bert(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
            token_type_ids=token_type_ids_flat
        )
        
        # 获取 [CLS] token 的 embedding: (Batch*12, 768)
        text_features = outputs.pooler_output
        # 恢复形状为 (Batch, 12, 768)
        text_features = text_features.view(batch_size, num_lines, -1)
        
        # 2. 应用 Line Mask (将 Padding 行的特征置零)
        if line_mask is not None:
            # line_mask: (Batch, 12) -> (Batch, 12, 1)
            mask = line_mask.unsqueeze(-1)
            
            # 确保 mask 与特征在同一设备
            mask = mask.to(text_features.device)
            
            text_features = text_features * mask
            visual = visual * mask
            acoustic = acoustic * mask
            
        # 3. 特征拼接 (Concatenation)
        # 拼接 Text, Acoustic, Visual
        # 结果形状: (Batch, 12, 768 + 128 + 768)
        combined_features = torch.cat([text_features, acoustic, visual], dim=-1)
        
        # 4. 展平 (Flatten)
        # 将 12 行的特征全部展平: (Batch, 12 * Total_Dim)
        combined_flat = combined_features.view(batch_size, -1)
        
        # 5. 预测 (Prediction)
        combined_flat = self.dropout(combined_flat)
        hidden = torch.relu(self.fc1(combined_flat))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        
        # 6. 计算损失 (Loss)
        if label_ids is not None:
            loss_fct = nn.L1Loss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            return loss
        else:
            return logits

    def test(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        line_mask=None,
        **kwargs
    ):
        # 复用 forward 逻辑，不传入 label_ids 以获取预测值
        return self.forward(
            input_ids, 
            visual, 
            acoustic, 
            attention_mask, 
            token_type_ids, 
            line_mask, 
            label_ids=None, 
            **kwargs
        )