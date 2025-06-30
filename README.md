# CM3060 NLP - Human Written Versus AI Generated Text Classification

## Project Overview

This project examines the rising difficulty of differentiating human-written text from AI generated texts, a task that becomes more significant with the advent of Large Language Models. The project performs a comparative examination of two key methods for text classification, a TF-IDF with Logistic Regression and an Deep Learning model utilizing the transformer architecture DistilBERT. The HC3 dataset, tailored for this task, is used to assess the performance of both models. Strategies for preprocessing, tokenization, and hyperparameter settings are examined thoroughly to ensure equitable comparison.

## Dataset: HC3

![](https://img.shields.io/badge/Languages-%20English-brightgreen) 
![](https://img.shields.io/badge/ChatGPT-grey)

- Paper: https://doi.org/10.48550/arXiv.2301.07597
- GitHub: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- Hugging Face: https://huggingface.co/datasets/Hello-SimpleAI/HC3
  
## Prerequisites

Install the required libraries with:
```
pip install datasets pandas scikit-learn nltk matplotlib seaborn transformers accelerate torch torchvision torchaudio
```
Or run the exact cells in the notebook.

Libraries Used:

- datasets (Hugging Face) – for dataset loading and management
- transformers (Hugging Face) – for tokenizer
- torch – PyTorch for Deep Learning model
- pandas, numpy – for data manipulation
- matplotlib, seaborn – for visualization
- nltk – for tokenization and lemmatization
- scikit-learn – for TF-IDF, logistic regression, and evaluation metrics

## Installation and Setup

1. Download or clone the repository:
```
git clone https://github.com/WZhengJie99/CM3060-NLP-Human-Written-Versus-AI-Generated-Text-Classification.git
```

2. Run the Jupyter Notebook.

## Conclusion Summary
														
![image](https://github.com/user-attachments/assets/b71b9d05-1a27-4882-b089-07d8b267c4ca)

In summary, the statistical model serves as a lightweight and interpretable model while the embedding-driven DL method provides slightly better gains in accuracy and resilience. Selection between both models should consider application limitations where simplicity and speed are essential, or in situations requiring greater accuracy and semantic insight. AI advancements have been growing continuously and especially so in recent years, going forward, future research directions would include datasets from various AI models that generate texts and dataset upkeeping to cater to the evolving AI models. Experimentation improvements would involve domain-specific variants or ensemble methods that combine different models to suit specific contexts.
