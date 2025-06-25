# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

# 确保其他脚本在同一目录下
from data_loader import BearingDataset
from model import BearingCLIP


def contrastive_loss(logits):
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.weights_dir, exist_ok=True)

    print("\n" + "=" * 20 + " Experiment Configuration " + "=" * 20)
    print(f"Training Mode: {args.mode}")
    print(f"Text Encoder Model: {args.text_model_name}")
    print(f"Prompt Template: '{args.prompt}'")
    print(f"File Suffix: '{args.save_suffix}'")
    print("=" * 66 + "\n")

    # 准备数据
    train_dataset = BearingDataset(data_dir=args.train_data_dir, prompt_template=args.prompt)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 定义模型
    model = BearingCLIP(
        embedding_dim=args.embedding_dim,
        use_adapter=(args.mode == 'adapter'),
        text_model_name=args.text_model_name
    ).to(device)

    # 设置训练模式、优化器和保存路径
    if args.mode == 'zero_shot':
        print("--- Starting Stage 1: Zero-Shot Pre-training ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        save_path = os.path.join(args.weights_dir, f'zero_shot_model{args.save_suffix}.pth')

    elif args.mode == 'adapter':
        print("--- Starting Stage 2: Adapter Fine-tuning ---")

        # 根据 save_suffix 智能加载对应的 zero-shot 模型
        base_model_name = f'zero_shot_model{args.save_suffix}.pth'
        load_path = os.path.join(args.weights_dir, base_model_name)

        print(f"Info: Adapter tuning will load base weights from: {load_path}")
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Prerequisite weights not found at {load_path}. Please run zero_shot training with the same --save_suffix first.")

        pretrained_dict = torch.load(load_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'adapter' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:
                print(f"Training adapter parameter: {name}")

        adapter_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(adapter_params, lr=args.lr)
        save_path = os.path.join(args.weights_dir, f'adapter_tuned_model{args.save_suffix}.pth')

    # 训练循环
    all_prompts = train_dataset.prompts
    for epoch in range(args.epochs):
        model.train()
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
            pbar.set_postfix({"Loss": loss.item()})

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bearing Diagnosis CLIP-like model.")
    parser.add_argument('--mode', type=str, required=True, choices=['zero_shot', 'adapter'])
    parser.add_argument('--train_data_dir', type=str, default='./bearingset/train_set/')
    parser.add_argument('--weights_dir', type=str, default='./pretrained_weights/')
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)

    # --- 用于消融实验的灵活参数 ---
    parser.add_argument('--prompt', type=str, default="A {} bearing", help="Prompt template.")
    parser.add_argument('--text_model_name', type=str, default='distilbert-base-uncased',
                        help="Hugging Face model name for the text encoder.")
    parser.add_argument('--save_suffix', type=str, default="",
                        help="A unique suffix for saved model files to manage experiments (e.g., '_distilbert_simple').")

    args = parser.parse_args()
    train(args)