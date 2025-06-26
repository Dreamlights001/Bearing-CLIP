# Bearing-CLIP: 基于CLIP思想的轴承故障诊断框架

本项目采用 CLIP的核心思想，将BERT模型创新性地应用于工业领域的轴承故障诊断。我们不再使用图像，而是将一维的振动时序信号与描述其状态的自然语言文本进行对齐，构建一个强大的“振动-文本”多模态诊断模型。

**CLIP** (Contrastive Language-Image Pre-Training) 在这里本身不是一个具体的模型，而是一种对比性的“训练思想”或“多模态学习框架”。它的核心思想是：

  - 双编码器结构 (Dual Encoders): 准备两个独立的编码器，一个用于处理一种模态的数据（例如图像），另一个用于处理另一种模态的数据（例如文本）。
  - 对比学习 (Contrastive Learning): 将大量的“数据-描述”对（例如“狗的图片 - a photo of a dog”）同时输入两个编码器，然后通过一个对比损失函数，在同一个高维空间中，将匹配的对拉近，将不匹配的对推远。
  - 目标: 最终得到一个对齐的、通用的多模态特征空间。在这个空间里，相似的概念（无论来自哪个模态）距离相近。

**BERT** (Bidirectional Encoder Representations from Transformers) 是一个具体、强大的“预训练语言模型”。它的核心任务是：

  - 理解文本: BERT是一个语言天才，它通过在海量的互联网文本上进行预训练，学会了语法、语义以及丰富的世界知识。
  - 生成文本特征: 它的专长是将输入的句子转换成高质量、蕴含丰富语义的数字向量（即特征向量或词嵌入）。
  - 该框架不仅实现了高精度的故障诊断，还内置了完善的消融实验流程，方便研究人员快速验证不同模型组件的有效性。

**通义千问2 (Qwen2)** 是由阿里巴巴达摩院在2024年推出的新一代开源大语言模型 (LLM)。作为业界领先的模型系列，它代表了当前自然语言处理领域的顶尖水平，并在性能、效率和开放性上都实现了重大突破。它的核心任务和优势在于：
  - 深度语义理解 (Deep Semantic Comprehension): 如果说BERT是一位语言天才，那么Qwen2则更像一位知识渊博、精通推理的领域专家。它通过在规模远超以往的高质量、多领域万亿级数据上进行预训练，并结合更先进的模型架构，使其不仅能理解语言的表层语法和语义，更能捕捉到文本之间极其细微的差别和深层的逻辑关联。

  - 生成顶尖的特征表示 (Generating State-of-the-Art Feature Representations): Qwen2的专长在于能将输入的文本（即使是简短的故障描述）转换成信息密度极高、区分度极强的特征向量。这种高质量的表示能力对于我们的诊断任务至关重要，因为它能帮助模型清晰地辨别“内圈故障”和“外圈故障”这类语义相近但物理意义完全不同的概念。

  - 卓越的模型扩展性与效率 (Excellent Scalability and Efficiency): Qwen2系列的一大亮点是提供了从0.5B（5亿）到72B（720亿）参数的全尺寸模型矩阵。这为研究和应用提供了极大的灵活性，允许我们根据具体的硬件条件和性能要求，选择最合适的模型（例如，我们实验中选用的Qwen2-1.5B就是在性能和资源消耗之间取得了绝佳平衡的型号）。
 
**这是一个基于CLIP思想的多模态故障诊断模型，其中文本编码部分我们采用了预训练的BERT和Qwen2模型。**

![](https://github.com/Dreamlights001/Bearing-CLIP/blob/main/images/Bearing-CLIP%E6%B5%81%E7%A8%8B%E5%9B%BE.svg)

## **✨ 核心功能**

* **CLIP多模态架构**: 创新性地将CLIP思想从视觉-语言领域迁移至工业信号处理，构建“振动-文本”双编码器模型。  
* **先进的两阶段训练流程**:  
  1. **零样本预训练**: 通过对比学习，建立振动信号模式与故障文本语义之间的深刻关联。  
  2. **适配器微调**: 引入轻量级残差适配器（Residual Adapter），实现参数高效（Parameter-Efficient）的微调，显著提升模型性能。  
* **强大的消融实验框架**:  
  * 通过命令行参数轻松切换和对比不同的**文本编码器**（例如，DistilBERT vs. 先进的 Qwen2）。  
  * 支持对比不同的**提示词模板**（例如，"A {} bearing" vs. "{}"）。  
  * 通过统一的脚本即可完成对**适配器有效性**的验证。  
* **灵活的训练与评估**:  
  * 通过 \--save\_suffix 参数为不同实验的产出模型进行**唯一命名和管理**。  
  * 通过 \--save\_path 参数将每次评估的结果（图表、报告、预测文件）保存到**独立的文件夹**，便于结果追踪和比较。  
* **全面的性能与特征可视化**:  
  * 自动计算并输出准确率、精确率、召回率、F1分数、AUROC等**多维度性能指标**。  
  * 生成带有详细分类指标的**混淆矩阵图**和直观的**t-SNE特征分布图**。  
* **一键生成报告级图表**: 内置 make\_diagrams\_matplotlib.py 脚本，可一键生成项目的工作流图，无需依赖外部软件。

## **📁 项目结构**
```
/  
├── bearingset/                 \# 数据集根目录  
├── pretrained\_weights/         \# 存放所有训练好的模型权重  
│   ├── zero\_shot\_model.pth  
│   ├── adapter\_tuned\_model.pth  
│   └── ... (消融实验产出的其他模型)  
├── results/                    \# 存放评估结果的文件夹  
│   ├── distilbert\_simple\_no\_adapter/  
│   └── ... (其他实验结果)  
├── workflow/                   \# 存放由脚本生成的流程图  
│   ├── 1\_main\_workflow\_mpl.png  
│   └── ...  
├── data\_loader.py              \# 数据加载模块  
├── model.py                    \# 模型定义模块 (支持加载不同Hugging Face模型)  
├── train.py                    \# 训练脚本 (支持多变量消融实验)  
├── evaluate.py                 \# 评估脚本 (支持多变量消融实验)  
├── make\_diagrams\_matplotlib.py \# (可选) 流程图生成脚本  
├── requirements.txt            \# Python依赖库  
└── README.md                   \# 本文档
```

## **🛠️ 安装与设置**

### **1\. 克隆仓库**

```Bash

git clone https://github.com/Dreamlights001/Bearing-CLIP.git  
cd Bearing-CLIP
```
### **2\. 创建并激活虚拟环境 (推荐)**

```Bash

python \-m venv venv  
# Windows  
venv\\Scripts\\activate  
# macOS / Linux  
source venv/bin/activate
```
### **3\. 安装依赖**

```Bash

pip install \-r requirements.txt
```
### **4\. 处理Hugging Face模型下载 (重要！)**

本项目需要从 Hugging Face 下载预训练的 `DistilBERT` 模型和 `Qwen/Qwen2-1.5B`模型。由于网络原因，可能会下载失败。请选择以下任一方法解决：

* 方法A: 设置Hugging Face镜像  
  在运行脚本的终端中，执行以下命令：  
  ```Bash  
  # Windows (CMD)  
  set HF\_ENDPOINT=https://hf-mirror.com  
  # Windows (PowerShell)  
  $env:HF\_ENDPOINT \= "https://hf-mirror.com"  
  # macOS / Linux  
  export HF\_ENDPOINT=https://hf-mirror.com

* 方法B: 手动下载模型  

  使用git将模型克隆到本地（例如，git clone https://hf-mirror.com/Qwen/Qwen2-1.5B-Base），然后在运行时将--text\_model\_name参数指向本地文件夹路径。
<p>

* 方法C(推荐): 程序自动集成  

### **5\. 准备数据集**

将您的轴承数据集放置在根目录下的 `bearingset` 文件夹中，并确保其子目录 `train_set` 和 `test_set` 包含 CSV 数据文件。

## **🚀 核心工作流：训练与评估最佳模型**

此流程使用默认的**完整提示词**和DistilBERT文本编码器。

### **阶段一: 零样本预训练**
此阶段使用默认的完整提示词模板进行训练。
```Bash

python train.py \--mode zero\_shot
```
* **产出**: ./pretrained\_weights/zero\_shot\_model.pth

### **阶段二: 适配器微调**
此阶段加载第一阶段的模型，并仅微调适配器。
```Bash

python train.py \--mode adapter \--epochs 15 \--lr 5e-5
```
* **产出**: ./pretrained\_weights/adapter\_tuned\_model.pth

### **阶段三: 评估最终模型**
评估脚本可以评估任何模型，并通过 `--save_path` 指定结果输出目录。
```Bash

python evaluate.py \--model\_path ./pretrained\_weights/adapter\_tuned\_model.pth \--save\_path ./results/final\_model\_report
```
* **产出**: 在 ./results/final\_model\_report 文件夹下生成完整的性能报告和图表。

## **🔬 高级用法: 进行消融实验**
本框架的核心优势在于其便捷的消融实验流程。
利用灵活的命令行参数，可以轻松完成复杂的对比实验。

### **实验1: 适配器 vs. 无适配器**

* **目的**: 验证残差适配器的有效性。  
* **运行**: 分别评估在核心工作流中产出的两个模型。
  ```Bash  
  # 评估无适配器模型  
  python evaluate.py \--model\_path ./pretrained\_weights/zero\_shot\_model.pth \--save\_path ./results/no\_adapter  
  # 评估有适配器模型  
  python evaluate.py \--model\_path ./pretrained\_weights/adapter\_tuned\_model.pth \--save\_path ./results/with\_adapter

* **分析**: 对比两个results文件夹中 `results/_no_adapter` 和 `results/_with_adapter` 的性能指标。

### **实验2: 文本编码器先进性对比 (DistilBERT vs. Qwen2)**

* **目的**: 验证使用更先进的LLM是否能提升性能。  
* **运行 (以完整提示词为例)**:  
  1. **训练并评估 DistilBERT**:  
     ```Bash  
     # 训练  
     python train.py \--mode zero\_shot \--text\_model\_name 'distilbert-base-uncased' \--prompt "A {} bearing" \--save\_suffix "\_distilbert\_simple"  
     python train.py \--mode adapter \--text\_model\_name 'distilbert-base-uncased' \--prompt "A {} bearing" \--save\_suffix "\_distilbert\_simple"  
     # 评估  
     python evaluate.py \--model\_path ./pretrained\_weights/adapter\_tuned\_model\_distilbert\_simple.pth \--text\_model\_name 'distilbert-base-uncased' \--save\_path ./results/distilbert\_simple\_with\_adapter

  2. **训练并评估 Qwen2**:  
     ```Bash  
     # 训练  
     python train.py \--mode zero\_shot \--text\_model\_name 'Qwen/Qwen2-1.5B-Base' \--prompt "A {} bearing" \--save\_suffix "\_qwen2\_simple"  
     python train.py \--mode adapter \--text\_model\_name 'Qwen/Qwen2-1.5B-Base' \--prompt "A {} bearing" \--save\_suffix "\_qwen2\_simple"  
     # 评估  
     python evaluate.py \--model\_path ./pretrained\_weights/adapter\_tuned\_model\_qwen2\_simple.pth \--text\_model\_name 'Qwen/Qwen2-1.5B-Base' \--save\_path ./results/qwen2\_simple\_with\_adapter

* **分析**: 对比两个最终的评估结果，观察LLM带来的性能变化。

### **实验 3: 提示词模板的重要性**
**目的**: 对比完整提示词 ("A {} bearing") 与简单标签 ("{}") 的效果。

**运行**: `train.py` 会根据 `--prompt` 参数自动处理文件名。

**训练和评估 (简单提示词)**:
```bash
# 训练 (会自动保存为 ..._simple.pth)
python train.py --mode zero_shot --text_model_name $model_name --prompt "{}" --save_suffix $save_suffix --epochs $epochs
python train.py --mode adapter --text_model_name $model_name --prompt "{}" --save_suffix $save_suffix --epochs $epochs

# 评估
python evaluate.py --model_path $pretrain --text_model_name $model_name --save_path $save_path
```

**训练和评估 (完整提示词)**:
使用核心工作流中已生成的 `adapter_tuned_model.pth` 进行评估，或重新运行不带 `--prompt` 参数的训练命令。

**分析**: 比较使用简单提示词和完整提示词的评估结果。

### **实验 4: 预训练语言模型的价值**
**目的**: 对比强大的 DistilBERT 和从零训练的 Simple-LSTM 文本编码器的效果。

**运行**: 使用 `--text_encoder_type` 参数。

**注意**：此实验需要对 `train.py` 和 `evaluate.py` 进行微调以管理不同的模型文件，或在训练后手动重命名模型文件。

## ⚙️ 模型框架细节
- **振动编码器**: 轻量级的一维卷积神经网络 (1D-CNN)。
- **文本编码器**: 预训练的 DistilBERT 或可切换的 Embedding+LSTM。
- **关键公式 - 相似度计算**:
<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Ctext%7Blogits%7D%20%3D%20%5Cexp%28%5Ctau%29%20%5Ccdot%20%28%5Ctext%7Bnormalize%7D%28V%29%20%5Ccdot%20%5Ctext%7Bnormalize%7D%28T%29%5ET%29" alt="Similarity Calculation">
</p>

- **关键公式 - 对称对比损失**:
<p align="center">
  <img src="https://latex.codecogs.com/png.latex?L_%7B%5Ctext%7Btotal%7D%7D%20%3D%20%5Cfrac%7B2%20%5Ccdot%20%5Ctext%7BCrossEntropy%7D%28L%2C%20%5Ctext%7Blabels%7D%29%20%2B%20%5Ctext%7BCrossEntropy%7D%28L%5ET%2C%20%5Ctext%7Blabels%7D%29%7D%7B2%7D" alt="Symmetric Contrastive Loss">
</p>

## **引用**

如果您的研究从本项目中受益，请考虑引用。

Code snippet
```
@misc{bearing-clip-framework,  
  author       \= {Your Name},  
  title        \= {Bearing-CLIP: A CLIP-Inspired Framework for Advanced Fault Diagnosis},  
  year         \= {2025},  
  publisher    \= {GitHub},  
  journal      \= {GitHub repository},  
  howpublished \= {\\url{https://github.com/Dreamlights001/Bearing-CLIP}}  
}
```
## **许可证**

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 许可证。

## **贡献**

**作者 王宇琛 吉林大学 机械与航空航天工程学院**<p>
**共同完成者 张植萱 吉林大学 计算机科学与技术学院**

## **启发**

本方法受到**[AnoVL(CVPR2023)](https://github.com/hq-deng/AnoVL)**的启发
