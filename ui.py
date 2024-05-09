import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import re
import json

# 加载模型和分词器
model_name = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 解析输入函数
def parse_input(input_text):
    matches = re.match(r'(\S+)\s+(\{.*?\})\s+(\S+)\s+(\S+)', input_text)
    if matches:
        keyword = matches.group(1)
        related_words = json.loads(matches.group(2).replace("'", "\""))
        predict_words = matches.group(3)
        category = matches.group(4)
        return keyword, related_words, predict_words, category
    else:
        raise ValueError("Invalid input format")

# 预测函数
def predict(input_mode, input_text, keyword, related_words, predict_words, category):
    if input_mode == "All-in-one":
        keyword, related_words, predict_words, category = parse_input(input_text)

    input_text = f"{keyword} {related_words} {predict_words} {category}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = probabilities.argmax().item()
    prediction_label = "不会点击" if predicted_class == 0 else "会点击"
    result_list = probabilities.numpy().tolist()[0]
    result_dict = {'不会点击' if k == 0 else '会点击': v for k, v in enumerate(result_list)}
    return prediction_label, result_dict

extra_info = """
### Note
Panelists: Jing Li, Guoyuan Li  
Model: bert-base-chinese  
Method: Fine-Tuning  
GitHub: [https://github.com/stepbystepcode/Click-Prediction](https://github.com/stepbystepcode/Click-Prediction)  
"""

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# Click Prediction Model")
    gr.Markdown("Enter the keyword, related words, predict words, and category to predict the click probability.")
    
    with gr.Row():
        with gr.Column():
            input_mode = gr.Radio(['All-in-one', 'Separate'], label='Input Mode', value='All-in-one')
            input_text = gr.Textbox(label="All In One")
            keyword = gr.Textbox(label="Keyword")
            related_words = gr.Textbox(label="Related Words")
            predict_words = gr.Textbox(label="Predict Words")
            category = gr.Textbox(label="Category")
        with gr.Column():
            prediction = gr.Label(label="Prediction")
            probabilities = gr.JSON(label="Probabilities")
            predict_button = gr.Button("Predict")
            extra_info_content = gr.Markdown(extra_info)

    def wrapped_predict(input_mode, input_text, keyword, related_words, predict_words, category):
        try:
            return predict(input_mode, input_text, keyword, related_words, predict_words, category)
        except ValueError as e:
            return str(e), {}

    predict_button.click(
        fn=wrapped_predict,
        inputs=[input_mode, input_text, keyword, related_words, predict_words, category],
        outputs=[prediction, probabilities]
    )

demo.launch(share=True)
