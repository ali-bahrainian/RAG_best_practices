import mauve
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch

import os
import json
from tqdm.contrib import tzip
from transformers import GPT2Tokenizer

def convert_numpy_types(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
    
def calculate_mauve_similarity(tokenizer, embedding_model, generated_answers, gold_answers):
    """
    Calculates cosine similarity between the generated and gold answers.

    Args:
        generated_answers (list[str]): Generated answers.
        gold_answers (list[str]): Gold standard answers.

    Returns:
        float: Cosine similarity between the answers.
    """
        # Generate embeddings
    generated_embeddings = embedding_model.encode(generated_answers)
    gold_embeddings = embedding_model.encode(gold_answers)

    # Calculate cosine similarity for each pair
    similarities = []
    for gen_emb in generated_embeddings:
        for gold_emb in gold_embeddings:
            similarity = mauve.compute_mauve(p_features=gen_emb, q_features=gold_emb, max_text_length=256)
            similarities.append(similarity)

    return similarities

def calculate_mauve(tokenizer, embedding_model, generated_answers, gold_answers):
    """
    Calculates Mauve score between the generated and gold answers.

    Args:
        generated_answers (list[str]): Generated answers.
        gold_answers (list[str]): Gold standard answers.

    Returns:
        float: Mauve score between the answers.
    """
        # Generate embeddings

    # Calculate cosine similarity for each pair
    similarities = []

    # gold_answers = [gold_answer for gold_answer in gold_answers if gold_answer]
    # gen_answers = [gen_answer]*len(gold_answers)
    # p_tokens = [torch.LongTensor(tokenizer.encode(gen_answer, return_tensors="pt")) for gen_answer in gen_answers]
    # q_tokens = [torch.LongTensor(tokenizer.encode(gold_answer, return_tensors="pt")) for gold_answer in gold_answers]
    # score = mauve.compute_mauve(p_tokens=p_tokens, q_tokens=q_tokens, device_id=0, max_text_length=256)    
    # print(p_tokens[0].shape)
    # print(gold_answers)
    # print(gen_answers)
    p_features = embedding_model.encode(generated_answers)
    q_features = embedding_model.encode(gold_answers)
    score = mauve.compute_mauve(p_features=p_features, q_features=q_features)
    print(score.mauve)
    similarities.append(score.mauve)

    return similarities

def calculate_metrics_corr(tokenizer, embedding_model, result):
    """
    Calculates and aggregates various metrics, including F1 scores and cosine similarities, for evaluation results.

    Args:
        result (dict): A dictionary representing the evaluation data and result for a single query.

    Returns:
        dict: A dictionary containing aggregated F1 scores and cosine similarities for different types of answers.
    """
    metrics = {}
    # r1f1_scores_dict = {}
    # r2f1_scores_dict = {}
    # rLf1_scores_dict = {}
    # similarities_dict = {}
    mauve_scores_dict = {}

    answer_type = 'correct'
    answers = [result['best_answer']]

    mauve_scores = calculate_mauve(tokenizer, embedding_model, [result['generated_response']], answers)

    mauve_scores_dict[f'mauve_{answer_type}'] = np.mean(mauve_scores)

    metrics.update(mauve_scores_dict)
    return metrics

def compute_evaluation_metrics(tokenizer, embedding_model, test_data):
    """
    Formats and combines evaluation results into a single DataFrame.

    Args:
        test_data (DataFrame): Original test data.

    Returns:
        DataFrame: Combined evaluation results.
    """
    data_and_evaluation = []
    gold_answers = test_data['best_answer'].tolist()
    generated_answers = test_data['generated_response'].tolist()
    
    mauve_score = calculate_mauve(tokenizer, embedding_model, generated_answers, gold_answers)

    return mauve_score
    
if __name__ == "__main__":
    # run = '13'
    # time = '04-08_17-18'

    # dir = f'./results/run{run}_{time}/'
    # dir_names = ['run1_03-12_20-11', 'run4_03-13_01-40', 'run11_03-25_11-34', 'run12_04-03_15-56', 'run20_04-29_12-29', 'run21_05-01_23-10', 'run22_09-02_16-28', 'run23_09-02_16-49', 'run25_09-09_00-19', 'run26_09-09_01-47', 'run26_09-09_10-43']
    dir_names = ['run13_04-08_17-18']
    
    ####### MMLU
    # dir_names = ['run1_08-24_12-30', 'run4_08-26_12-12', 'run11_09-04_08-31', 'run12_08-30_09-04', 'run13_08-30_12-41', 'run19_09-10_12-09', 'run20_09-05_14-03', 'run21_09-03_10-15', 'run22_09-05_21-12', 'run23_09-05_20-46']  # , 'run25_09-09_01-01', 'run26_09-09_01-47', 'run27_09-09_00-59'
    # dir_names = ['run27_09-11_01-15', 'run27_09-14_15-40']
    # dir_names = ['run27_09-14_15-40']
    for dir in dir_names:
        # dir = './results_mmlu/'+dir
        dir = './results/'+dir
        all_entries = os.listdir(dir)
        files_ = [entry for entry in all_entries if os.path.isfile(os.path.join(dir, entry))]
        names = [filename.replace('evaluation_', '').replace('.pkl', '') for filename in files_ if filename.startswith('evaluation')]
        print(names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_dfs = [pd.read_pickle(f'{dir}/evaluation_{name}.pkl') for name in names]
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_model = SentenceTransformer(embedding_model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        
        # metrics = ['r1f1', 'r2f1', 'rLf1', 'similarity', 'mauve']
        metrics = ['mauve']
        # Initialize the dictionary to store computed means
        computed_means = {name: {metric: {} for metric in metrics} for name in names}
        
        for name, df in tzip(names, eval_dfs):
            print(name)
            mauve_score = compute_evaluation_metrics(tokenizer, embedding_model, df)
            computed_means[name]['mauve'] = mauve_score
                
            print(computed_means[name])
        computed_means = convert_numpy_types(computed_means)

        with open(f"{dir}/mauve_results.json", "w") as outfile: 
            json.dump(computed_means, outfile)    
        