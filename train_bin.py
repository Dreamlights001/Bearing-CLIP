# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
from transformers import DistilBertTokenizer  # 为简单文本编码器准备

# 确保其他脚本在同一目录下
from data_loader import BearingDataset
from model import BearingCLIP


def contrastive_loss(logits):
    # 对称交叉熵损失
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.weights_dir, exist_ok=True)

    # --- 1. 自动根据 prompt 生成文件后缀 ---
    if args.prompt == '{}':
        save_suffix = "_simple"
        print(f"Info: Simple prompt template detected. Models will be saved with '{save_suffix}' suffix.")
    else:
        save_suffix = ""
        print(f"Info: Full prompt template detected. Models will be saved with default names.")

    # --- 2. 准备数据 ---
    print(f"Using prompt template: '{args.prompt}'")
    train_dataset = BearingDataset(data_dir=args.train_data_dir, prompt_template=args.prompt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # --- 3. 定义模型 ---
    # 为消融实验准备的逻辑
    simple_tokenizer = None
    vocab_size = None
    if args.text_encoder_type == 'simple_lstm':
        simple_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        vocab_size = simple_tokenizer.vocab_size

    model = BearingCLIP(
        embedding_dim=args.embedding_dim,
        use_adapter=(args.mode == 'adapter'),
        text_encoder_type=args.text_encoder_type,
        vocab_size=vocab_size,
        simple_tokenizer=simple_tokenizer
    ).to(device)

    # --- 4. 设置训练模式、优化器和保存路径 ---
    if args.mode == 'zero_shot':
        print("\n--- Starting Stage 1: Zero-Shot Pre-training ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        save_path = os.path.join(args.weights_dir, f'zero_shot_model{save_suffix}.pth')

    elif args.mode == 'adapter':
        print("\n--- Starting Stage 2: Adapter Fine-tuning ---")

        # 自动推断需要加载的 zero-shot 模型路径
        base_model_name = f'zero_shot_model{save_suffix}.pth'
        load_path = os.path.join(args.weights_dir, base_model_name)

        print(f"Info: Automatically loading base weights from: {load_path}")
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Prerequisite weights not found at {load_path}. Please run zero_shot training for this prompt template first.")

        pretrained_dict = torch.load(load_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'adapter' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # 冻结除适配器外的所有参数
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:
                print(f"Training adapter parameter: {name}")

        adapter_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(adapter_params, lr=args.lr)
        save_path = os.path.join(args.weights_dir, f'adapter_tuned_model{save_suffix}.pth')

    # --- 5. 训练循环 ---
    all_prompts = train_dataset.prompts

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for vibration_batch, label_indices in pbar:
            vibration_batch = vibration_batch.to(device)
            text_batch = [all_prompts[i] for i in label_indices]

            optimizer.zero_grad()

            vibration_features, text_features, logit_scale = model(vibration_batch, text_batch)

            logits = logit_scale * vibration_features @ text_features.T

            loss = contrastive_loss(logits)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # --- 6. 保存模型 ---
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bearing Diagnosis CLIP-like model.")
    parser.add_argument('--mode', type=str, required=True, choices=['zero_shot', 'adapter'],
                        help="Training mode: 'zero_shot' (pre-training) or 'adapter' (fine-tuning).")
    parser.add_argument('--train_data_dir', type=str, default='./bearingset/train_set/', help="Path to training data.")
    parser.add_argument('--weights_dir', type=str, default='./pretrained_weights/',
                        help="Directory to save model weights.")

    parser.add_argument('--embedding_dim', type=int, default=512, help="Dimension of the embedding space.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")

    # --- 消融实验相关参数 ---
    parser.add_argument('--prompt', type=str, default="A {} bearing",
                        help="Prompt template to use for text encoding. Use '{}' as placeholder for the label.")
    parser.add_argument('--text_encoder_type', type=str, default='distilbert', choices=['distilbert', 'simple_lstm'],
                        help="Type of text encoder to use. Relevant for ablation studies.")

    args = parser.parse_args()
    train(args)