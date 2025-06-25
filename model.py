# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# --- 关键修复：确保下面这行 import 存在且正确 ---
from transformers import AutoModel, AutoTokenizer
import numpy as np


# (SimpleTextEncoder, ResidualAdapter, VibrationEncoder 的定义保持不变，为完整性一并提供)

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.tokenizer = None

    def forward(self, text_list):
        if not self.tokenizer: raise ValueError("SimpleTextEncoder's tokenizer must be set before use.")
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.embedding.weight.device)
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        return hidden.squeeze(0)


class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.ReLU(),
                                     nn.Linear(bottleneck_dim, input_dim))

    def forward(self, x): return x + self.adapter(x)


class VibrationEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.conv_net = nn.Sequential(nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=31), nn.ReLU(),
                                      nn.BatchNorm1d(16), nn.MaxPool1d(kernel_size=2, stride=2),
                                      nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15), nn.ReLU(),
                                      nn.BatchNorm1d(32), nn.MaxPool1d(kernel_size=2, stride=2),
                                      nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7), nn.ReLU(),
                                      nn.BatchNorm1d(64), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv_net(x);
        x = x.view(x.size(0), -1);
        return self.fc(x)


# --- TextEncoder 的定义 ---
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=512, model_name='distilbert-base-uncased'):
        super().__init__()
        print(f"Initializing TextEncoder with model: {model_name}")

        trust_code = 'Qwen' in model_name or 'deepseek' in model_name or 'ChatGLM' in model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_code)
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=trust_code)

        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)

        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, text_list):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(self.bert.device) for key, val in inputs.items()}
        bert_output = self.bert(**inputs).last_hidden_state
        sentence_embedding = bert_output.mean(dim=1)
        text_embedding = self.projection(sentence_embedding)
        return text_embedding


# --- BearingCLIP 的定义 ---
class BearingCLIP(nn.Module):
    def __init__(self, embedding_dim=512, use_adapter=False,
                 text_encoder_type='distilbert',
                 text_model_name='distilbert-base-uncased',
                 vocab_size=None, simple_tokenizer=None):
        super().__init__()
        self.vibration_encoder = VibrationEncoder(embedding_dim)

        if text_encoder_type == 'simple_lstm':
            # 兼容之前的消融实验
            from transformers import DistilBertTokenizer
            if vocab_size is None:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                vocab_size = tokenizer.vocab_size
                simple_tokenizer = tokenizer
            self.text_encoder = SimpleTextEncoder(vocab_size, embedding_dim)
            if simple_tokenizer is not None:
                self.text_encoder.tokenizer = simple_tokenizer
        else:
            self.text_encoder = TextEncoder(embedding_dim, model_name=text_model_name)

        self.adapter = None
        if use_adapter:
            self.adapter = ResidualAdapter(embedding_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, vibration_data, text_list):
        vibration_features = self.vibration_encoder(vibration_data)

        if self.adapter:
            vibration_features = self.adapter(vibration_features)

        text_features = self.text_encoder(text_list)

        vibration_features = F.normalize(vibration_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        return vibration_features, text_features, self.logit_scale.exp()