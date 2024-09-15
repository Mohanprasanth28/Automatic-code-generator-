import torch
from transformers import BartForConditionalGeneration, BartTokenizerFast
from torch.utils.data import Dataset, DataLoader
import csv
import json

class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tokenized_data[idx]['input_ids']),
            'attention_mask': torch.tensor(self.tokenized_data[idx]['attention_mask']),
            'labels': torch.tensor(self.tokenized_data[idx]['labels'])
        }

# Function to tokenize descriptions and codes
def tokenize_data(csv_file, tokenizer, max_source_length, max_target_length):
    # Initialize lists to store tokenized data
    tokenized_data = []

    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            description = row[0]
            code = row[1]

            # Tokenize description and code
            inputs = tokenizer(description, truncation=True, max_length=max_source_length, padding="max_length", return_tensors='pt')
            labels = tokenizer(code, truncation=True, max_length=max_target_length, padding="max_length", return_tensors='pt')['input_ids']

            # Add tokenized data to list
            tokenized_data.append({
                'input_ids': inputs['input_ids'].squeeze().tolist(),
                'attention_mask': inputs['attention_mask'].squeeze().tolist(),
                'labels': labels.squeeze().tolist()
            })

    return tokenized_data

# Paths and Hyperparameters
csv_file = ' """set your dataset path"""  '
json_tokenized_file = ' """ set your model token path"""  '
model_output_dir = ' """ model path""" '

model_name = 'facebook/bart-base'
max_source_length = 128
max_target_length = 128
batch_size = 8
num_epochs = 20
learning_rate = 1e-5

# Initialize BART tokenizer and model
tokenizer = BartTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Tokenize data and save to JSON file
tokenized_data = tokenize_data(csv_file, tokenizer, max_source_length, max_target_length)
with open(json_tokenized_file, 'w') as json_file:
    json.dump(tokenized_data, json_file)

# Create Custom Dataset from tokenized JSON file
dataset = CustomDataset(tokenized_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model to train mode
model.train()

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Progressive Unfreezing
unfreeze_steps = [0.3, 0.6, 1.0]  # Unfreeze at 30%, 60%, and 100% of total batches

# Training Loop
total_batches = len(dataloader)
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_num, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Progressive Unfreezing
        if (batch_num + 1) / total_batches in unfreeze_steps:
            unfreeze_layers = int((batch_num + 1) / total_batches * len(model.model.encoder.layers))  # Accessing encoder layers correctly
            for param in model.parameters():
                param.requires_grad = False
            for param in model.model.encoder.layers[-unfreeze_layers:].parameters():  # Accessing encoder layers correctly
                param.requires_grad = True
            for param in model.model.decoder.layers[-unfreeze_layers:].parameters():  # Accessing decoder layers correctly
                param.requires_grad = True

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

# Save the trained model and tokenizer
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

print("Training complete. Model saved at:", model_output_dir)
