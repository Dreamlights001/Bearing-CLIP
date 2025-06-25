#DistilBERT (基准模型)

#简单模版
#训练 (无适配器 - Zero-Shot)
python train.py --mode zero_shot --text_model_name 'distilbert-base-uncased' --prompt "{}" --save_suffix "_distilbert_simple" --epochs 30
#训练 (有适配器 - Adapter-Tuned)
python train.py --mode adapter --text_model_name 'distilbert-base-uncased' --prompt "{}" --save_suffix "_distilbert_simple" --epochs 15 --lr 5e-5
#评估 (无适配器)
python evaluate.py --model_path ./pretrained_weights/zero_shot_model_distilbert_simple.pth --text_model_name 'distilbert-base-uncased' --save_path ./results/distilbert_simple_no_adapter
#评估 (有适配器)
python evaluate.py --model_path ./pretrained_weights/adapter_tuned_model_distilbert_simple.pth --text_model_name 'distilbert-base-uncased' --save_path ./results/distilbert_simple_with_adapter

#正常模版
#训练 (无适配器 - Zero-Shot)
python train.py --mode zero_shot --text_model_name 'distilbert-base-uncased' --prompt "A {} bearing" --save_suffix "_distilbert" --epochs 30
#训练 (有适配器 - Adapter-Tuned)
python train.py --mode adapter --text_model_name 'distilbert-base-uncased' --prompt "A {} bearing" --save_suffix "_distilbert" --epochs 15 --lr 5e-5
#评估 (无适配器)
python evaluate.py --model_path ./pretrained_weights/zero_shot_model_distilbert.pth --text_model_name 'distilbert-base-uncased' --save_path ./results/distilbert_no_adapter
#评估 (有适配器)
python evaluate.py --model_path ./pretrained_weights/adapter_tuned_model_distilbert.pth --text_model_name 'distilbert-base-uncased' --save_path ./results/distilbert_with_adapter


#Qwen2-1.5B (更新模型)

#简单模版
#训练 (无适配器 - Zero-Shot)
python train.py --mode zero_shot --text_model_name 'Qwen/Qwen2-1.5B' --prompt "{}" --save_suffix "_qwen2_simple" --epochs 30
#训练 (有适配器 - Adapter-Tuned)
python train.py --mode adapter --text_model_name 'Qwen/Qwen2-1.5B' --prompt "{}" --save_suffix "_qwen2_simple" --epochs 15 --lr 5e-5
#评估 (无适配器)
python evaluate.py --model_path ./pretrained_weights/zero_shot_model_qwen2_simple.pth --text_model_name 'Qwen/Qwen2-1.5B' --save_path ./results/qwen2_simple_no_adapter
#评估 (有适配器)
python evaluate.py --model_path ./pretrained_weights/adapter_tuned_model_qwen2_simple.pth --text_model_name 'Qwen/Qwen2-1.5B' --save_path ./results/qwen2_simple_with_adapter

#正常模版
#训练 (无适配器 - Zero-Shot)
python train.py --mode zero_shot --text_model_name 'Qwen/Qwen2-1.5B' --prompt "A {} bearing" --save_suffix "_qwen2" --epochs 30
#训练 (有适配器 - Adapter-Tuned)
python train.py --mode adapter --text_model_name 'Qwen/Qwen2-1.5B' --prompt "A {} bearing" --save_suffix "_qwen2" --epochs 15 --lr 5e-5
#评估 (无适配器)
python evaluate.py --model_path ./pretrained_weights/zero_shot_model_qwen2.pth --text_model_name 'Qwen/Qwen2-1.5B' --save_path ./results/qwen2_no_adapter
#评估 (有适配器)
python evaluate.py --model_path ./pretrained_weights/adapter_tuned_model_qwen2.pth --text_model_name 'Qwen/Qwen2-1.5B' --save_path ./results/qwen2_with_adapter
