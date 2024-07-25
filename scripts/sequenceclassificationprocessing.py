import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import glob

# List of CSV files
csv_files = glob.glob("path_to_csv/*.csv")

# Initialize an empty DataFrame
df = pd.DataFrame()

# Read and concatenate all CSV files
for csv_file in csv_files:
    temp_df = pd.read_csv(csv_file)
    df = pd.concat([df, temp_df], ignore_index=True)

# Filter rows where is_article is True
df = df[df['is_article'] == True]

# Extract relevant information
data = df[['all_text', 'x', 'y', 'width', 'height']].to_dict(orient='records')

# Function to create pairs and labels based on position
def create_pairs_and_labels_position(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if (data[i]['y'] < data[j]['y']) or (data[i]['x'] < data[j]['x']):
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
            else:
                pairs.append({"article1": data[j]['all_text'], "article2": data[i]['all_text'], "label": 0})
    return pairs

# Function to create pairs and labels based on size
def create_pairs_and_labels_size(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            size_i = data[i]['width'] * data[i]['height']
            size_j = data[j]['width'] * data[j]['height']
            if size_i > size_j:
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
            else:
                pairs.append({"article1": data[j]['all_text'], "article2": data[i]['all_text'], "label": 0})
    return pairs

# Function to create pairs and labels based on image presence
def create_pairs_and_labels_image(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            # Assuming 'has_image' column exists in the data
            if data[i]['has_image'] and not data[j]['has_image']:
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
            elif not data[i]['has_image'] and data[j]['has_image']:
                pairs.append({"article1": data[j]['all_text'], "article2": data[i]['all_text'], "label": 0})
            else:
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
    return pairs

# Create pairs and labels based on different criteria
pairs_position = create_pairs_and_labels_position(data)
pairs_size = create_pairs_and_labels_size(data)
pairs_image = create_pairs_and_labels_image(data)

# Convert to DataFrame
df_position = pd.DataFrame(pairs_position)
df_size = pd.DataFrame(pairs_size)
df_image = pd.DataFrame(pairs_image)

# Convert the DataFrames to Hugging Face Datasets
dataset_position = Dataset.from_pandas(df_position)
dataset_size = Dataset.from_pandas(df_size)
dataset_image = Dataset.from_pandas(df_image)

# Initialize a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define a function to concatenate articles and tokenize them
def tokenize_function(examples):
    return tokenizer(
        examples['article1'] + tokenizer.sep_token + examples['article2'], 
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Apply the tokenize function to the datasets
tokenized_dataset_position = dataset_position.map(tokenize_function, batched=True)
tokenized_dataset_size = dataset_size.map(tokenize_function, batched=True)
tokenized_dataset_image = dataset_image.map(tokenize_function, batched=True)

# Load a pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define a trainer for position-based training
trainer_position = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_position,
    eval_dataset=tokenized_dataset_position
)

# Define a trainer for size-based training
trainer_size = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_size,
    eval_dataset=tokenized_dataset_size
)

# Define a trainer for image presence-based training
trainer_image = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_image,
    eval_dataset=tokenized_dataset_image
)

# Train the models
trainer_position.train()
trainer_size.train()
trainer_image.train()
