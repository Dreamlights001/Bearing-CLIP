# evaluate.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import os
import argparse
from tqdm import tqdm
import pandas as pd

# 从其他文件导入需要的类
from data_loader import BearingDataset
from model import BearingCLIP


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    class_metrics = report_df.loc[class_names]
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    table_text = "Class-wise Metrics:\n\n" + class_metrics.to_string(float_format="{:0.4f}".format)
    plt.figtext(0.5, -0.1, table_text, ha="center", fontsize=11, family="monospace",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}, transform=plt.gcf().transFigure)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    cm_path = os.path.join(save_path, 'confusion_matrix_with_metrics.png')
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"\nConfusion matrix with detailed metrics saved to {cm_path}")
    plt.close()


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.save_path, exist_ok=True)

    test_dataset = BearingDataset(data_dir=args.test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- MODIFIED: 将 text_model_name 参数传递给模型 ---
    use_adapter = 'adapter' in args.model_path
    model = BearingCLIP(
        embedding_dim=args.embedding_dim,
        use_adapter=use_adapter,
        text_model_name=args.text_model_name  # <-- 此处使用了新参数
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found at {args.model_path}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path} with text encoder '{args.text_model_name}'")

    all_prompts = test_dataset.prompts
    with torch.no_grad():
        text_features_all = model.text_encoder(all_prompts).to(device)
        text_features_all = torch.nn.functional.normalize(text_features_all, p=2, dim=-1)

    all_preds, all_labels, all_probs, all_vibration_features = [], [], [], []
    with torch.no_grad():
        for vibration_batch, label_indices in tqdm(test_loader, desc="Evaluating"):
            vibration_batch = vibration_batch.to(device)
            vibration_features = model.vibration_encoder(vibration_batch)
            if model.adapter:
                vibration_features = model.adapter(vibration_features)
            vibration_features = torch.nn.functional.normalize(vibration_features, p=2, dim=-1)
            similarity = (100.0 * vibration_features @ text_features_all.T).softmax(dim=-1)
            probs, preds = similarity.max(dim=1)
            all_preds.extend(preds.cpu().numpy());
            all_labels.extend(label_indices.cpu().numpy())
            all_probs.extend(similarity.cpu().numpy());
            all_vibration_features.append(vibration_features.cpu().numpy())

    all_vibration_features = np.concatenate(all_vibration_features, axis=0)
    class_names = test_dataset.unique_labels
    print("\n" + "=" * 25 + " Model Performance " + "=" * 25)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n--- 1. Overall Accuracy ---\nAccuracy: {accuracy:.4f}")
    print("\n--- 2. Classification Report (Precision, Recall, F1-Score) ---")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    print("\n--- 3. Area Under the Receiver Operating Characteristic Curve (AUROC) ---")
    all_labels_binarized = label_binarize(all_labels, classes=range(len(class_names)))
    if all_labels_binarized.shape[1] == len(class_names) and len(np.unique(all_labels)) > 1:
        auroc = roc_auc_score(all_labels_binarized, all_probs, multi_class='ovr', average='weighted')
        print(f"Weighted Average AUROC: {auroc:.4f}")
    else:
        print("AUROC calculation skipped: Not all classes were present or only one class present in the test set.")
    print("\n--- 4. Confusion Matrix & t-SNE Visualization ---")
    plot_confusion_matrix(all_labels, all_preds, class_names, args.save_path)
    print("\nGenerating t-SNE plot...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vibration_features) - 1))
    tsne_features = tsne.fit_transform(all_vibration_features)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=[class_names[i] for i in all_labels],
                    palette='deep', legend='full')
    plt.title('t-SNE Visualization of Vibration Features', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12);
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    tsne_path = os.path.join(args.save_path, 'tsne_visualization.png')
    plt.savefig(tsne_path)
    print(f"t-SNE plot saved to {tsne_path}")
    plt.close()
    pred_labels_text = [class_names[p] for p in all_preds]
    output_path = os.path.join(args.save_path, 'predictions.txt')
    with open(output_path, 'w') as f:
        for label in pred_labels_text: f.write(f"{label}\n")
    print(f"Predictions saved to {output_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Bearing Diagnosis model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model weights (.pth file).")
    parser.add_argument('--save_path', type=str, default='./result/', help="Directory to save evaluation results.")
    parser.add_argument('--test_data_dir', type=str, default='./bearingset/test_set/', help="Path to test data.")
    parser.add_argument('--embedding_dim', type=int, default=512, help="Dimension of the embedding space.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for evaluation.")

    # --- MODIFIED: 添加缺失的命令行参数定义 ---
    parser.add_argument('--text_model_name', type=str, default='distilbert-base-uncased',
                        help="The pre-trained model name from Hugging Face for the text encoder.")
    # (为了兼容性，保留旧的text_encoder_type，但在此工作流中不起作用)
    parser.add_argument('--text_encoder_type', type=str, default='distilbert', choices=['distilbert', 'simple_lstm'])

    args = parser.parse_args()
    evaluate(args)