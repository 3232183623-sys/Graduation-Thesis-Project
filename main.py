# 安装必备库
!pip install -q transformers==4.45.0 datasets pillow tqdm matplotlib

# 4bit 量化所需库
!pip install -q bitsandbytes accelerate

from google.colab import drive
drive.mount('/content/drive')

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import random
from datasets import load_dataset
import os

# 加载 RH-BENCH 数据集的训练集（使用 'train' 拆分）
dataset = load_dataset("LCZZZZ/RH-Bench", split="train")

# 将数据集转换为列表，才能进行随机抽样
dataset_list = list(dataset)

# 随机抽取 10 张图片
random_sample = random.sample(dataset_list, 10)
print(f"Using {len(random_sample)} samples for testing.")  # 输出样本数

# 重新加载图片路径
def load_image_from_path(path: str) -> Image.Image:
    """加载图片并转换为 RGB"""
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    else:
        print(f"Warning: Image not found at {path}. Skipping.")
        return None

# 使用轻量的 BLIP-2 模型（Salesforce/blip2-opt-2.7b）
model_id = "Salesforce/blip2-opt-2.7b"  # 轻量版 BLIP-2 模型
model = AutoModelForVision2Seq.from_pretrained(model_id).to("cuda")

# 加载处理器
processor = AutoProcessor.from_pretrained(model_id)

# 推理函数（简化了 token 数量）
def blip2_inference(model, processor, image: Image.Image, question: str) -> str:
    """
    使用 BLIP-2 模型进行推理
    """
    full_prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(
        text=full_prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)
    # 限制推理长度为较短的 token 数量
    output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    answer = processor.decode(output_ids[0], skip_special_tokens=True)
    answer = answer.split("ASSISTANT:")[-1].strip()
    return answer

# 评测函数，计算准确率与幻觉率
def evaluate_baseline(inference_fn, model, processor, dataset):
    """
    运行基线评测：计算准确率和幻觉数
    """
    correct = 0
    hallucinations = 0
    total = len(dataset)
    for data in tqdm(dataset):
        image_rel_path = data['image']
        img_path = f"/content/drive/MyDrive/MLRM-Halu/RH-Bench/{image_rel_path}"
        print(f"Loading image from: {img_path}")  # 输出加载路径
        img = load_image_from_path(img_path)
        if img is None:
            continue  # 跳过没有找到的图像
        question = data['question']
        pred = inference_fn(model, processor, img, question)

        # 判断是否正确
        if pred.strip().lower() == data['answer'].strip().lower():
            correct += 1
        else:
            hallucinations += 1
    accuracy = correct / total if total > 0 else 0
    hallucination_rate = hallucinations / total if total > 0 else 0
    return accuracy, hallucination_rate, total

# 调用评测函数并打印结果
accuracy, hallucination_rate, total = evaluate_baseline(blip2_inference, model, processor, random_sample)
print(f"Total samples: {total}, Accuracy: {accuracy:.2f}, Hallucination rate: {hallucination_rate:.2f}")

# 简单的结果可视化（展示准确率与幻觉率）
reasoning_lengths = [32]  # 只考虑简化后的推理长度
baseline_accuracy = [accuracy]
baseline_hallucination = [hallucination_rate]

plt.figure()
plt.plot(reasoning_lengths, baseline_accuracy, marker='o', label='Accuracy')
plt.xlabel('Reasoning length (tokens)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Reasoning Length')
plt.legend()
plt.show()

plt.figure()
plt.plot(reasoning_lengths, baseline_hallucination, marker='o', label='Hallucination Rate')
plt.xlabel('Reasoning length (tokens)')
plt.ylabel('Hallucination Rate')
plt.title('Hallucination Rate vs Reasoning Length')
plt.legend()
plt.show()
