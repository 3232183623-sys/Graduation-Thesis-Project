# 多模态幻觉现象实证分析项目

## 项目描述
本项目用于验证和分析多模态大模型（LLaVA）中的幻觉现象，并测试一种轻量化的缓解策略。

## 环境配置
1. 安装依赖: `pip install -r requirements.txt`
2. （可选）设置模型缓存路径以避免重复下载，详见代码中的 `model_cache_dir` 变量。

## 运行方式
直接运行主文件即可: `python main.py`

## 运行结果
最后会得到一个近似与原论文的结果
<img width="2000" height="1200" alt="hallucination_rate_vs_reasoning_length_limited" src="https://github.com/user-attachments/assets/c12c51fa-e402-4cd2-a4d5-30d2153e94e6" />
折线图：推理链长度与幻觉率的关系
下图展示了 **推理链长度** 与 **幻觉率** 之间的关系。可以看出，推理链长度增加时，幻觉率逐步下降，证明短推理链有助于减少幻觉现象的发生。

<img width="2000" height="1200" alt="accuracy_vs_reasoning_length_limited" src="https://github.com/user-attachments/assets/22aeff30-06be-4e19-b617-6302bf34ef2b" />
折线图：推理链长度与准确率的关系
此图展示了不同推理链长度下模型 **准确率** 的变化趋势。可以看出，推理链长度与准确率呈现正相关关系，随着推理链的增加，准确率有稳定的提升。
