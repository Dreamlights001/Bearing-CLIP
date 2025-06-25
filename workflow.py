# make_diagrams_matplotlib.py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def setup_matplotlib_font():
    """设置matplotlib支持中文黑体显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        print("字体已设置为 'SimHei' (黑体)。如果显示为方块，请确保您的系统中已安装此字体。")
    except Exception as e:
        print(f"设置字体失败: {e}。将使用默认字体。")


def draw_box(ax, xy, text, boxstyle="round,pad=0.5", facecolor='#F5F5F5', edgecolor='black', **kwargs):
    """在指定坐标绘制一个带文本的方框"""
    ax.text(xy[0], xy[1], text,
            ha='center', va='center',
            bbox=dict(boxstyle=boxstyle, fc=facecolor, ec=edgecolor, lw=1),
            **kwargs)


def draw_arrow(ax, start_xy, end_xy, label=""):
    """在两个坐标点之间绘制一个带标签的箭头"""
    arrowprops = dict(arrowstyle="->,head_width=0.4,head_length=0.8",
                      shrinkA=5, shrinkB=5, fc="black", ec="black",
                      connectionstyle="arc3,rad=0")
    ax.annotate("", xy=end_xy, xytext=start_xy, arrowprops=arrowprops)
    if label:
        mid_xy = ((start_xy[0] + end_xy[0]) / 2, (start_xy[1] + end_xy[1]) / 2)
        ax.text(mid_xy[0], mid_xy[1] + 0.1, label, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="white", alpha=0.7))


def create_main_workflow(directory):
    """流程图一：主线训练与评估工作流"""
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    title = "流程图一：主线训练与评估工作流"
    fig.suptitle(title, fontsize=20, y=0.98)

    pos = {
        'start': (5, 9.2), 'stage1': (5, 7.5), 'output1': (5, 6),
        'stage2': (5, 4.5), 'output2': (5, 3), 'eval': (5, 1.5), 'results': (5, 0)
    }
    texts = {
        'start': "开始: 项目设置",
        'stage1': '''阶段一: 零样本预训练\n使用完整的提示词模板，学习通用的振动-文本表示。\n参数设置:\n- mode: 'zero_shot'\n- prompt: "A {} bearing"\n- epochs: 30\n- lr: 1e-4''',
        'output1': "产出:\nzero_shot_model.pth",
        'stage2': '''阶段二: 适配器微调\n加载预训练权重，仅微调轻量级残差适配器。\n参数设置:\n- mode: 'adapter'\n- epochs: 15\n- lr: 5e-5''',
        'output2': "产出:\nadapter_tuned_model.pth",
        'eval': '''最终模型评估\n在测试集上验证最终模型的性能。\n参数设置:\n- batch_size: 256''',
        'results': '''最终结果\n1. 性能报告 (准确率/AUROC等)\n2. 混淆矩阵图\n3. t-SNE可视化图\n4. 预测结果文件'''
    }

    draw_box(ax, pos['start'], texts['start'], facecolor='#D6EAF8')
    draw_box(ax, pos['stage1'], texts['stage1'], fontsize=9, facecolor='#EBF5FB')
    draw_box(ax, pos['output1'], texts['output1'], boxstyle="round,pad=0.6", facecolor='#E3F2FD')
    draw_box(ax, pos['stage2'], texts['stage2'], fontsize=9, facecolor='#EBF5FB')
    draw_box(ax, pos['output2'], texts['output2'], boxstyle="round,pad=0.6", facecolor='#E3F2FD')
    draw_box(ax, pos['eval'], texts['eval'], fontsize=9, facecolor='#EBF5FB')
    # --- MODIFIED: Replaced 'document' with a valid style 'round' ---
    draw_box(ax, pos['results'], texts['results'], boxstyle="round,pad=0.5", fontsize=9, facecolor='#E8F5E9')

    draw_arrow(ax, (5, 8.8), (5, 8.2), "准备数据集和环境")
    draw_arrow(ax, (5, 6.8), (5, 6.4), "执行 `python train.py ...`")
    draw_arrow(ax, (5, 5.6), (5, 5.2), "加载基础权重")
    draw_arrow(ax, (5, 3.8), (5, 3.4), "执行 `python train.py ...`")
    draw_arrow(ax, (5, 2.6), (5, 2.0), "加载最终模型")
    draw_arrow(ax, (5, 1.0), (5, 0.5), "执行 `python evaluate.py ...`")

    plt.savefig(os.path.join(directory, '1_main_workflow_mpl.png'), bbox_inches='tight')
    plt.close(fig)
    print("已生成: 1_main_workflow_mpl.png")


def create_ablation_workflow(directory):
    """流程图二：消融实验工作流"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    title = "流程图二：消融实验工作流 (以提示词模板为例)"
    fig.suptitle(title, fontsize=20, y=0.98)

    pos = {
        'hypothesis': (5, 9), 'train_a': (2.5, 7), 'eval_a': (2.5, 5), 'res_a': (2.5, 3),
        'train_b': (7.5, 7), 'eval_b': (7.5, 5), 'res_b': (7.5, 3), 'compare': (5, 1.5), 'conclusion': (5, 0)
    }
    texts = {
        'hypothesis': '提出假设:\n完整的提示词是否比简单标签更好?',
        'train_a': '训练基准模型\n参数: --prompt "A {} bearing"',
        'eval_a': '评估基准模型', 'res_a': '性能结果 A', 'train_b': '训练实验模型\n参数: --prompt "{}"',
        'eval_b': '评估实验模型', 'res_b': '性能结果 B', 'compare': '对比分析\n比较结果A和结果B的各项性能指标',
        'conclusion': '得出结论\n例如: "完整提示词使准确率提升了X%"'
    }

    draw_box(ax, pos['hypothesis'], texts['hypothesis'], facecolor='#FADBD8')
    draw_box(ax, pos['train_a'], texts['train_a'], facecolor='#EBF5FB')
    draw_box(ax, pos['eval_a'], texts['eval_a'], facecolor='#EBF5FB')
    # --- MODIFIED: Replaced 'document' with a valid style 'round' ---
    draw_box(ax, pos['res_a'], texts['res_a'], boxstyle="round,pad=0.5", facecolor='#E8F5E9')
    draw_box(ax, pos['train_b'], texts['train_b'], facecolor='#FEF9E7')
    draw_box(ax, pos['eval_b'], texts['eval_b'], facecolor='#FEF9E7')
    # --- MODIFIED: Replaced 'document' with a valid style 'round' ---
    draw_box(ax, pos['res_b'], texts['res_b'], boxstyle="round,pad=0.5", facecolor='#E8F5E9')
    draw_box(ax, pos['compare'], texts['compare'], boxstyle="sawtooth", facecolor='#D2B4DE')
    draw_box(ax, pos['conclusion'], texts['conclusion'], boxstyle="round4", facecolor='#D5F5E3')

    draw_arrow(ax, (3, 8.5), (2.5, 7.5));
    draw_arrow(ax, (7, 8.5), (7.5, 7.5))
    draw_arrow(ax, pos['train_a'], pos['eval_a']);
    draw_arrow(ax, pos['eval_a'], pos['res_a'])
    draw_arrow(ax, pos['train_b'], pos['eval_b']);
    draw_arrow(ax, pos['eval_b'], pos['res_b'])
    draw_arrow(ax, pos['res_a'], pos['compare']);
    draw_arrow(ax, pos['res_b'], pos['compare'])
    draw_arrow(ax, pos['compare'], pos['conclusion'])

    plt.savefig(os.path.join(directory, '2_ablation_workflow_mpl.png'), bbox_inches='tight')
    plt.close(fig)
    print("已生成: 2_ablation_workflow_mpl.png")


def create_inference_workflow(directory):
    """流程图三：模型推理细节工作流"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    title = "流程图三：模型推理细节工作流"
    fig.suptitle(title, fontsize=20, y=0.98)

    pos = {
        'start': (6, 9.2), 'labels': (2, 7.5), 'prompts': (2, 6.2), 'text_encoder': (2, 4.5), 'text_db': (2, 2.5),
        'vib_sample': (10, 7.5), 'vib_encoder': (10, 6), 'adapter': (10, 4.7), 'vib_feature': (10, 3.4),
        'compare': (6, 2.5), 'argmax': (6, 1.2), 'aggregate': (6, 0), 'final_metrics': (9, 0)
    }
    texts = {
        'start': "开始: 运行 evaluate.py", 'labels': '所有类别标签\n["health", "inner_race_fault", ...]',
        'prompts': '生成所有文本提示',
        'text_encoder': '文本编码器 (Text Encoder)\n- DistilBERT (预训练)\n- 线性投影层',
        'text_db': '所有类别的文本特征库 T_all',
        'vib_sample': '单个振动信号样本 V_test',
        'vib_encoder': '振动编码器 (Vibration Encoder)\n- 1D 卷积神经网络 (CNN)',
        'adapter': '残差适配器 (可选)', 'vib_feature': '单个样本的振动特征 v_test',
        'compare': '计算相似度 (核心步骤)\nscores = normalize(v_test) @ normalize(T_all).T',
        'argmax': '选择最高分 (Argmax)\n找到相似度分数最高的类别索引', 'aggregate': '收集所有预测\n与真实标签',
        'final_metrics': '输出最终性能报告'
    }

    draw_box(ax, pos['start'], texts['start'], facecolor='#D6EAF8')
    draw_box(ax, pos['labels'], texts['labels'], facecolor='#FEF9E7')
    draw_box(ax, pos['prompts'], texts['prompts'], facecolor='#FEF9E7')
    draw_box(ax, pos['text_encoder'], texts['text_encoder'], fontsize=9, facecolor='#FCF3CF')
    draw_box(ax, pos['text_db'], texts['text_db'], boxstyle="round,pad=0.6", facecolor='#E3F2FD')
    draw_box(ax, pos['vib_sample'], texts['vib_sample'], facecolor='#EBF5FB')
    draw_box(ax, pos['vib_encoder'], texts['vib_encoder'], fontsize=9, facecolor='#DDECF7')
    draw_box(ax, pos['adapter'], texts['adapter'], facecolor='#DDECF7')
    draw_box(ax, pos['vib_feature'], texts['vib_feature'], facecolor='#DDECF7')
    draw_box(ax, pos['compare'], texts['compare'], fontsize=9, facecolor='#F5B7B1')
    draw_box(ax, pos['argmax'], texts['argmax'], facecolor='#D2B4DE')
    draw_box(ax, pos['aggregate'], texts['aggregate'], facecolor='lightgrey')
    # --- MODIFIED: Replaced 'document' with a valid style 'round' ---
    draw_box(ax, pos['final_metrics'], texts['final_metrics'], boxstyle="round,pad=0.5", facecolor='#E8F5E9')

    draw_arrow(ax, pos['labels'], pos['prompts']);
    draw_arrow(ax, pos['prompts'], pos['text_encoder'])
    draw_arrow(ax, pos['text_encoder'], pos['text_db']);
    draw_arrow(ax, pos['vib_sample'], pos['vib_encoder'])
    draw_arrow(ax, pos['vib_encoder'], pos['adapter']);
    draw_arrow(ax, pos['adapter'], pos['vib_feature'])
    draw_arrow(ax, pos['text_db'], (pos['compare'][0] - 1, pos['compare'][1]))
    draw_arrow(ax, pos['vib_feature'], (pos['compare'][0] + 1, pos['compare'][1]))
    draw_arrow(ax, pos['compare'], pos['argmax']);
    draw_arrow(ax, pos['argmax'], pos['aggregate'])
    draw_arrow(ax, pos['aggregate'], pos['final_metrics'], "遍历结束后")

    plt.savefig(os.path.join(directory, '3_inference_workflow_mpl.png'), bbox_inches='tight')
    plt.close(fig)
    print("已生成: 3_inference_workflow_mpl.png")


if __name__ == '__main__':
    setup_matplotlib_font()
    output_directory = 'workflow'
    os.makedirs(output_directory, exist_ok=True)
    create_main_workflow(output_directory)
    create_ablation_workflow(output_directory)
    create_inference_workflow(output_directory)
    print(f"\n所有流程图已成功保存在 '{output_directory}' 文件夹下。")