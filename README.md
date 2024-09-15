# BART Code Generator

This project leverages Facebook's BART model (Bidirectional and Auto-Regressive Transformers) to generate code from natural language descriptions. The application is designed to streamline the process of code generation by transforming human-readable descriptions into functional code snippets. The project uses PyTorch and Hugging Face's Transformers library and includes a Flask web interface.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Training Process](#training-process)
- [Model Deployment](#model-deployment)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project aims to generate programming code from natural language descriptions using a sequence-to-sequence model, BART. The project includes a web interface where users can input descriptions and receive corresponding code outputs.

## Features
- **Natural Language Processing**: Leverages the BART transformer model to translate human-readable descriptions into code.
- **Progressive Unfreezing**: Utilizes a training method where layers of the model are gradually unfrozen for fine-tuning.
- **Web Interface**: Allows for user-friendly interaction through a web-based interface built with Flask.
- **Ngrok Integration**: Exposes the local Flask server to the internet using Ngrok.

## Setup
To set up and run this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/BART-Code-Generator.git
    cd BART-Code-Generator
    ```

2. **Install the required dependencies**:
    ```bash
    pip install torch transformers flask flask-ngrok pyngrok
    ```

3. **Download the pre-trained BART model**:
    If you have a pre-trained BART model, save it in the desired path or load a model like `facebook/bart-base` as a base.

4. **Set up Ngrok**:
    Create an Ngrok account, get your token, and set it up:
    ```bash
    ngrok authtoken YOUR_NGROK_TOKEN
    ```

5. **Prepare the Dataset**:
    - Format the dataset as a CSV file with two columns: `description` and `code`.
    - Update the path in the script to point to your dataset.

## Training Process
The model is trained using descriptions as inputs and the corresponding code snippets as target outputs. Progressive unfreezing is employed during training to gradually fine-tune the model layers. The training loop handles the tokenization of data, batch processing, and model optimization using AdamW optimizer.

1. **Run Training Script**:
    Update the script with your dataset and paths, and execute it to begin training:
    ```bash
    python train_model.py
    ```

    Model checkpoints will be saved in the specified directory.

## Model Deployment
Once training is complete, you can deploy the model using Flask and Ngrok.

1. **Start Flask Server**:
    ```bash
    python app.py
    ```

2. **Access the Web Interface**:
    Use the Ngrok-generated public URL to interact with the web app.

## How to Use
- Navigate to the Flask app via the Ngrok public URL.
- Enter a natural language description of the code you want.
- Press **Submit** to generate the corresponding code.
- The generated code will be displayed on the page.

## Contributing
We welcome contributions! Please follow these steps if you'd like to contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`.
3. Make your changes and commit them.
4. Push to the branch: `git push origin feature-branch-name`.
5. Submit a pull request.

## License
This project is licensed under the MIT License.
