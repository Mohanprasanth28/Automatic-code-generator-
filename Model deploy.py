import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok  # Import ngrok

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)  # Run ngrok when app runs
app.static_folder = 'static'

# Load the trained model and tokenizer
model = BartForConditionalGeneration.from_pretrained(' """give your model path"""  ')
tokenizer = BartTokenizer.from_pretrained(' """give your model path""" ')

# Function to generate code from description
def generate_code(description):
    input_ids = tokenizer(description, return_tensors='pt')['input_ids']
    output = model.generate(input_ids, max_length=500, num_beams=4, early_stopping=False)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        description = request.form['description']
        generated_code = generate_code(description)
        return render_template('index.html', generated_code=generated_code)
    return render_template('index.html', generated_code="")

if __name__ == "__main__":
    app.run()  # This will run with ngrok automatically
