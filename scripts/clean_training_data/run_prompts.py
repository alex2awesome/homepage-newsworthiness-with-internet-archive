from datasets import load_from_disk
import pandas as pd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import unicodedata
import os, json
import torch
import logging
import random


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
BATCH_SIZE = 500

def load_model(model_name: str):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(
        model_name,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        max_model_len=60_000
    )
    return tokenizer, model

def write_to_file(fname, ids, outputs):
    with open(fname, 'wb') as file:
        for id, output in zip(ids, outputs):
            response = output.outputs[0].text
            response = unicodedata.normalize('NFKC', response)
            if response and id:
                output = {}
                output['id'] = str(id)
                output['response'] = str(response)
                file.write(json.dumps(output).encode('utf-8'))
                file.write(b'\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # meta-llama/Llama-3.1-8B-Instruct
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--input_data_file', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--id_col', type=str, default='index')
    parser.add_argument('--prompt_col', type=str, default='prompt')
    parser.add_argument('--output_file', type=str, default=None)

    args = parser.parse_args()
    article_df = pd.read_csv(args.input_data_file, index_col=0)

    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(article_df)
    
    # load the model
    sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)
    tokenizer, model = load_model(args.model)
    num_batches = (args.end_idx - args.start_idx) // BATCH_SIZE
    batch_indices = [(i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, args.end_idx)) for i in range(num_batches)]
    random.shuffle(batch_indices)

    for start_idx, end_idx in tqdm(batch_indices):
        dirname = os.path.dirname(args.output_file)
        if (dirname != '') and not os.path.exists(dirname):
            os.makedirs(dirname)

        out_dirname, out_fname = os.path.split(args.output_file)
        fname, fext = os.path.splitext(out_fname)
        output_fname = f'{out_dirname}/{fname}__{start_idx}_{end_idx}{fext}' if out_dirname else f'{fname}__{start_idx}_{end_idx}{fext}'
        if not os.path.exists(output_fname):
            logging.info(f"Running prompts for batch {start_idx} to {end_idx}")
            with open(output_fname, 'w') as f:
                f.write('')

            df = article_df.iloc[start_idx:end_idx]
            clean_prompts = df[args.prompt_col].tolist()
            cleaned_article_outputs = model.generate(clean_prompts, sampling_params)
            write_to_file(output_fname, df[args.id_col], cleaned_article_outputs)
        else:
            logging.info(f"Skipping batch {start_idx} to {end_idx} as it already exists")