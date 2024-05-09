import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# 读取txt文件
with open('1000-点击预测.txt', 'r', encoding='utf-8') as file1:
    lines1 = file1.readlines()
with open('500-test.txt', 'r', encoding='utf-8') as file2:
    lines2 = file2.readlines()

# 解析数据
data1 = []
for line in lines1:
    parts = line.strip().split('\t')
    if len(parts) == 5:
        keyword, related_words, predict_words, category, click = parts
        data1.append((keyword, related_words, predict_words, category, int(click)))
data2 = []
for line in lines1:
    parts = line.strip().split('\t')
    if len(parts) == 5:
        keyword, related_words, predict_words, category, click = parts
        data2.append((keyword, related_words, predict_words, category, int(click)))

# 划分训练集和验证集
train_data = data1
val_data = data2

# 加载预训练模型和tokenizer
model_name = "bert-base-chinese"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = []
    labels = []
    for keyword, related_words, predict_words, category, click in examples:
        input_text = f"{keyword} {related_words} {predict_words} {category}"
        inputs.append(input_text)
        labels.append(click)
    tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=128)
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': labels
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 准备训练数据
train_dataset = Dataset.from_dict(preprocess_function(train_data))

# 准备验证数据
val_dataset = Dataset.from_dict(preprocess_function(val_data))

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=300,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.0001,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=100,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 保存微调后的模型
tokenizer.save_pretrained('./fine_tuned_model')
trainer.save_model('./fine_tuned_model')

# 训练结束后进行评估
eval_results = trainer.evaluate()

# 输出评估结果
print("Evaluation results:", eval_results)
