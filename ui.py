import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_name = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict(keyword, related_words, predict_words, category):
    # Prepare the input text in the same format as during training
    input_text = f"{keyword} {related_words} {predict_words} {category}"
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    # Get predictions from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    # Get the predicted class (0 or 1)
    predicted_class = probabilities.argmax().item()
    # Return the predicted class and probabilities of each class
    return {"Prediction": predicted_class, "Probability": probabilities.numpy().tolist()}

# Define the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Keyword"),
        gr.Textbox(label="Related Words"),
        gr.Textbox(label="Predict Words"),
        gr.Textbox(label="Category")
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.JSON(label="Probabilities")
    ],
    title="Click Prediction Model",
    description="Enter the keyword, related words, predict words, and category to predict the click probability."
)

# Launch the interface
iface.launch()
