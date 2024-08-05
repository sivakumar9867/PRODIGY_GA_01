# Import necessary libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define your custom dataset
# This dataset should be a list of text samples
custom_dataset = [
    "This is a sample text.",
    "Another sample text.",
    "More sample text here."
]

# Convert the list of texts into a Dataset object
dataset = Dataset.from_dict({"text": custom_dataset})

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize the text samples, padding and truncating to a fixed length
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Apply the tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory where the model predictions and checkpoints will be written.
    evaluation_strategy="epoch",     # Evaluation strategy to use. We are evaluating at the end of each epoch.
    learning_rate=2e-5,              # The initial learning rate for the optimizer.
    per_device_train_batch_size=4,   # Batch size per device during training.
    num_train_epochs=1,              # Total number of training epochs.
    weight_decay=0.01                # Strength of weight decay.
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # The model to train.
    args=training_args,                  # Arguments to pass to the Trainer.
    train_dataset=tokenized_dataset      # The dataset to use for training.
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("fine_tuned_gpt2")
tokenizer.save_pretrained("fine_tuned_gpt2")

# Function to generate text using the fine-tuned model
def generate_text(prompt):
    """
    Generates text from a prompt using the fine-tuned GPT-2 model.

    Args:
        prompt (str): The prompt to use for text generation.

    Returns:
        str: The generated text.
    """
    # Encode the prompt into input IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text from the input IDs
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text

# Example usage of the text generation function
if _name_ == "_main_":
    # Generate text based on a prompt
    prompt = "The quick brown fox"
    generated_text = generate_text(prompt)
    
    # Print the generated text
    print(generated_text)