import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm.auto import tqdm
import concurrent.futures
from multiprocessing import cpu_count
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to read a CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"{file_path} is empty. Skipping.")
            return None
        if 'is_article' not in df.columns:
            logger.error(f"'is_article' column not found in {file_path}. Skipping.")
            return None
        return df
    except pd.errors.EmptyDataError:
        logger.warning(f"{file_path} is empty or corrupted. Skipping.")
        return None

# Functions to create pairs and labels
def create_pairs_and_labels_position(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if (data[i]['y'] < data[j]['y']) or (data[i]['x'] < data[j]['x']):
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
            else:
                pairs.append({"article1": data[j]['all_text'], "article2": data[i]['all_text'], "label": 0})
    return pairs

def create_pairs_and_labels_size(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            size_i = data[i]['width'] * data[i]['height']
            size_j = data[j]['width'] * data[j]['height']
            if size_i > size_j:
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
            else:
                pairs.append({"article1": data[j]['all_text'], "article2": data[i]['all_text'], "label": 0})
    return pairs

'''def create_pairs_and_labels_image(data):
    pairs = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i].get('has_image', False) and not data[j].get('has_image', False):
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
            elif not data[i].get('has_image', False) and data[j].get('has_image', False):
                pairs.append({"article1": data[j]['all_text'], "article2": data[i]['all_text'], "label": 0})
            else:
                pairs.append({"article1": data[i]['all_text'], "article2": data[j]['all_text'], "label": 1})
    return pairs'''

# Directory paths for train and test files
train_dir = "../../html-bb-jpg-samples/"
test_dir = "../../html-bb-jpg-samples/"

# List of train and test CSV files
train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]

# Initialize a tokenizer
logger.info("Initializing tokenizer")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize articles
def tokenize_function(examples):
    return tokenizer(
        str(examples['article1']) + tokenizer.sep_token + str(examples['article2']), 
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Function to process each CSV file
def process_file_pair(df, df2, create_pairs_fn):
    data = df[['all_text', 'x', 'y', 'width', 'height']].to_dict(orient='records') + df2[['all_text', 'x','y', 'width', 'height']].to_dict(orient='records')
    pairs = create_pairs_fn(data)
    pairs_df = pd.DataFrame(pairs)
    dataset = Dataset.from_pandas(pairs_df)
    tokenized_dataset = dataset.map(tokenize_function, batched=False, num_proc=max(1, min(10, cpu_count() - 1)))
    return tokenized_dataset
logger.info("Used cpu " + str(cpu_count()))
# Process each train and test file independently
logger.info("Processing train and test files")
train_datasets_position = []
train_datasets_size = []
train_datasets_image = []

test_datasets_position = []
test_datasets_size = []
test_datasets_image = []

temp_train_files = []
for k in train_files:
	df = read_csv_file(k)
	if df is not None:
		df = df[df['is_article'] == True]
		if not df.empty:
			temp_train_files.append(df)
train_files = temp_train_files
logger.info(type(temp_train_files[0]))
logger.info(type(train_files[0]))
def process_file_pair_parallel(pair):
    position = process_file_pair(pair[0], pair[1], create_pairs_and_labels_position)
    size = process_file_pair(pair[0], pair[1], create_pairs_and_labels_size)
    return position, size

# This function handles the results and appends them to the datasets
def handle_results(result):
    position, size = result
    if position is not None:
        train_datasets_position.append(position)
    if size is not None:
        train_datasets_size.append(size)
batch_size = 1000  # Adjust this based on your memory constraints
total_pairs = [(curr_file, other_file) for curr_file in train_files for other_file in train_files]
total = len(total_pairs)
batches = [total_pairs[i:i + batch_size] for i in range(0, total, batch_size)]

for batch in tqdm(batches):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, int(cpu_count()/2))) as executor:
        futures = [executor.submit(process_file_pair_parallel, pair) for pair in batch]

        for future in concurrent.futures.as_completed(futures):
            handle_results(future.result())

# Concatenate the tokenized datasets
logger.info("Concatenating tokenized datasets")
train_dataset_position = Dataset.from_concat(train_datasets_position)
train_dataset_size = Dataset.from_concat(train_datasets_size)
#train_dataset_image = Dataset.from_concat(train_datasets_image)

test_dataset_position = Dataset.from_concat(test_datasets_position)
test_dataset_size = Dataset.from_concat(test_datasets_size)
#test_dataset_image = Dataset.from_concat(test_datasets_image)

# Load a pre-trained model
logger.info("Loading pre-trained model")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
logger.info("Defining training arguments")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize trainers
logger.info("Initializing trainers")
trainer_position = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_position,
    eval_dataset=test_dataset_position
)

trainer_size = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_size,
    eval_dataset=test_dataset_size
)

'''trainer_image = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_image,
    eval_dataset=test_dataset_image
)'''

# Training function with progress bar
def train_model(trainer, description):
    progress_bar = tqdm(range(training_args.num_train_epochs), desc=description)
    for _ in progress_bar:
        trainer.train()
        # Evaluation at the end of each epoch
        eval_metrics = trainer.evaluate()
        progress_bar.set_postfix({'eval_loss': eval_metrics['eval_loss']})
    progress_bar.close()

# Train the models with progress bars
logger.info("Training position-based model")
train_model(trainer_position, "Training position-based model")

logger.info("Training size-based model")
train_model(trainer_size, "Training size-based model")

#logger.info("Training image presence-based model")
#train_model(trainer_image, "Training image presence-based model")

logger.info("Training complete")

