"""
用预训练的Bert模型微调IMDB数据集，并使用SwanLabCallback回调函数将结果上传到SwanLab。
IMDB数据集的1是positive，0是negative。
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from swanlab.integration.huggingface import SwanLabCallback
import swanlab

def predict(text, model, tokenizer, CLASS_NAME):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

    print(f"Input Text: {text}")
    print(f"Predicted class: {int(predicted_class)} {CLASS_NAME[int(predicted_class)]}")
    return int(predicted_class)

# 加载IMDB数据集
dataset = load_dataset('imdb')

# 加载预训练的BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 定义tokenize函数
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# 对数据集进行tokenization
tokenized_datasets = dataset.map(tokenize, batched=True)

# 设置模型输入格式
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# 加载预训练的BERT模型
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_first_step=100,
    # 总的训练轮数
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
    # 单卡训练
)

CLASS_NAME = {0: "negative", 1: "positive"}

# 设置swanlab回调函数
swanlab_callback = SwanLabCallback(project='BERT',
                                   experiment_name='BERT-IMDB',
                                   config={'dataset': 'IMDB', "CLASS_NAME": CLASS_NAME})

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    callbacks=[swanlab_callback],
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')

# 测试模型
test_reviews = [
    "I absolutely loved this movie! The storyline was captivating and the acting was top-notch. A must-watch for everyone.",
    "This movie was a complete waste of time. The plot was predictable and the characters were poorly developed.",
    "An excellent film with a heartwarming story. The performances were outstanding, especially the lead actor.",
    "I found the movie to be quite boring. It dragged on and didn't really go anywhere. Not recommended.",
    "A masterpiece! The director did an amazing job bringing this story to life. The visuals were stunning.",
    "Terrible movie. The script was awful and the acting was even worse. I can't believe I sat through the whole thing.",
    "A delightful film with a perfect mix of humor and drama. The cast was great and the dialogue was witty.",
    "I was very disappointed with this movie. It had so much potential, but it just fell flat. The ending was particularly bad.",
    "One of the best movies I've seen this year. The story was original and the performances were incredibly moving.",
    "I didn't enjoy this movie at all. It was confusing and the pacing was off. Definitely not worth watching."
]

model.to('cpu')
text_list = []
for review in test_reviews:
    label = predict(review, model, tokenizer, CLASS_NAME)
    text_list.append(swanlab.Text(review, caption=f"{label}-{CLASS_NAME[label]}"))

if text_list:
    swanlab.log({"predict": text_list})

swanlab.finish()