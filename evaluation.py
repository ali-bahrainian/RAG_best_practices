import gc
import json
import os
import argparse
from datetime import datetime
import random

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset

from model.index_builder import IndexBuilder
from model.language_model import LanguageModel
from model.model_loader import ModelLoader
from model.rag import RAG
from model.retriever import Retriever

from config import configs_run1, configs_run2 

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the RAG model")
    parser.add_argument('--dataset', default='truthfulqa', type=str, help='Dataset to evaluate on') 
    parser.add_argument('--output-dir', default='outputs', type=str, help='Output directory')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    return parser.parse_args()

# Set random seed
def set_random_seed(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch for CPU
    torch.cuda.manual_seed(seed)  # PyTorch for GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    
    # This ensures that your code will be as deterministic as possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Initialize index builder
def initialize_index_builder(knowledge_base, config):
    index_builder = IndexBuilder(knowledge_base, config['embedding_model_name'], config['ralm']['expand_query'], **config['index_builder'])
    return index_builder.initialize_components()

# Initialize RAG model
def initialize_rag(knowledge_base, config, model_loader_generation, model_loader_seq2seq, index_pre, same_index, first_run):
    build_index = not same_index or first_run

    # Initialize index builder if needed
    if build_index:
        index, index_titles, doc_info = initialize_index_builder(knowledge_base, config)
    else:
        index, index_titles, doc_info = index_pre[0], index_pre[1], index_pre[2]
    retriever = Retriever(index, doc_info, config['embedding_model_name'], model_loader_seq2seq, index_titles)
    language_model = LanguageModel(model_loader_generation, config['is_chat_model'], config['instruct_tokens'])
    if not same_index:
        del index, index_titles, doc_info
        gc.collect()
        index_pre = None
    else:
        index_pre = (index, index_titles, doc_info)
    return RAG(retriever, language_model, **config['ralm']), index_pre

    
def mean_metrics_item(evaluation):
    metrics = ['r1f1','r2f1','rLf1', 'similarity', 'mauve']
        
    # Initialize the dictionary to store computed means
    computed_means = {}

    # Compute means and populate the dictionary
    for metric in metrics:
        computed_means[metric] = float(evaluation[metric].mean())

    return computed_means
    
if __name__ == "__main__":

    args = parse_args()
    set_random_seed(args.seed)

    if args.dataset == 'truthfulqa':
        truthful_qa = load_dataset("truthful_qa", "generation", split='validation').to_pandas()
        test_data = truthful_qa[['question', 'best_answer', 'correct_answers', 'incorrect_answers']]
        test_data['correct_answers'] = test_data['correct_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
        test_data['correct_answers'] = test_data['correct_answers'].apply(lambda x: [i for i in x if i]) # Remove empty strings from correct answers
        test_data['incorrect_answers'] = test_data['incorrect_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
        test_data['incorrect_answers'] = test_data['incorrect_answers'].apply(lambda x: [i for i in x if i])
        test_data = test_data[(test_data['correct_answers'].apply(len) > 1) & (test_data['incorrect_answers'].apply(len) > 1)]
        test_data = test_data.reset_index(drop=True)

    elif args.dataset == 'mmlu':
        mmlu = load_dataset("cais/mmlu", "all")
        test_data = mmlu['test'].to_pandas().groupby('subject').head(32).drop(columns='subject').reset_index(drop=True)
        def extract_answers(row):
            best_answer = row['choices'][row['answer']]
            incorrect_answers = np.array([choice for i, choice in enumerate(row['choices']) if i != row['answer']])
            return pd.Series([best_answer, incorrect_answers], index=['best_answer', 'incorrect_answers'])
        test_data[['best_answer', 'incorrect_answers']] = test_data.apply(extract_answers, axis=1)
        test_data = test_data.drop(columns=['choices', 'answer'])
        test_data['correct_answers'] = [np.array([]) for _ in range(len(test_data))]
        test_data = test_data[['question', 'best_answer', 'correct_answers', 'incorrect_answers']]

    
    knowledge_base = pd.read_pickle('resources/articles_l3.pkl')
    all_results = {}

    # Evaluate all configurations
    for configs, run in zip([configs_run1, configs_run2],[1, 2]):
        time = datetime.now().strftime("%m-%d_%H-%M")
        results_dir = f'{args.output_dir}/run{run}_{time}'

        os.makedirs(results_dir, exist_ok=True)
        index_configs = [c['index_builder'] for c in configs.values()]
        same_index = all(ic == index_configs[0] for ic in index_configs)
        index_pre = None
        first_run = True
        
        evaluations = {}
        for name, config in configs.items():
            # Initialize model loaders
            model_loader_generation = ModelLoader(config['generation_model_name'], 'causal', quant_type='4bit')
            model_loader_seq2seq = ModelLoader(config['seq2seq_model_name'], 'seq2seq', quant_type='4bit')
            
            # Load knowledge base
            if config['ralm']['icl_kb']:
                kb = test_data
            elif config['ralm']['kb_10K']:
                kb = pd.read_pickle('./resources/articles_l4.pkl')
            else:
                kb = knowledge_base
            
            ralm, index_pre = initialize_rag(kb, config, model_loader_generation, model_loader_seq2seq, index_pre, same_index, first_run)
            print(f"Evaluating model: {name}")
            evaluations[name] = ralm.evaluate(test_data)
        
            del ralm
            del model_loader_generation
            del model_loader_seq2seq
            gc.collect()
            torch.cuda.empty_cache()
            first_run = False
            
            # Save evaluation results
            evaluations[name].to_pickle(os.path.join(results_dir, f'evaluation_{name}.pkl'))
            with open(os.path.join(results_dir, f'config_{name}.json'), 'w') as f:
                json.dump(configs[name], f, indent=4)
        
            results = mean_metrics_item(evaluations[name])

            with open(f"{results_dir}/eval_results_{name}.json", "w") as outfile: 
                json.dump(results, outfile)   
            all_results[name] = results
        del index_pre
                
        with open(f"{results_dir}/eval_results_all.json", "w") as outfile: 
            json.dump(results, outfile)   
