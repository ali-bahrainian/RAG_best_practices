import pandas as pd
import numpy as np

from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# import mauve

import os
import json
from tqdm.contrib import tzip

def show_resultdf(eval_dfs):
    print(names)
    for i in range(len(eval_dfs[0])):
        print('========input_text========')
        print(eval_dfs[0][['input_text', 'generated_response']].loc[i,'input_text'])
        print('-------best_answer-------')
        print(eval_dfs[0].loc[i,'best_answer'])
        print('### generated_response ###')
        print(eval_dfs[0][['input_text', 'generated_response']].loc[i,'generated_response'])
    print(eval_dfs[0].columns)
    
def compute_evaluation_metrics(embedding_model, test_data):
    """
    Formats and combines evaluation results into a single DataFrame.

    Args:
        test_data (DataFrame): Original test data.

    Returns:
        DataFrame: Combined evaluation results.
    """
    data_and_evaluation = []
    for qid in test_data.index:
        result = test_data.loc[qid].to_dict()
        result.update(calculate_metrics_corr(embedding_model, result))

        data_and_evaluation.append(result)

    return pd.DataFrame(data_and_evaluation)

def calculate_metrics_corr(embedding_model, result):
    """
    Calculates and aggregates various metrics, including F1 scores and cosine similarities, for evaluation results.

    Args:
        result (dict): A dictionary representing the evaluation data and result for a single query.

    Returns:
        dict: A dictionary containing aggregated F1 scores and cosine similarities for different types of answers.
    """
    metrics = {}
    r1f1_scores_dict = {}
    r2f1_scores_dict = {}
    rLf1_scores_dict = {}
    similarities_dict = {}
    mauve_scores_dict = {}

    answer_type = 'correct'
    answers = result[f'{answer_type}_answers']
    answers = answers.tolist() + [result['best_answer']]*2

    # Calculate F1 scores and similarities
    r1f1_scores = calculate_f1_score([result['generated_response']], answers, n='1')
    r2f1_scores = calculate_f1_score([result['generated_response']], answers, n='2')
    rLf1_scores = calculate_f1_score([result['generated_response']], answers, n='L')
    similarities = calculate_cosine_similarity(embedding_model, [result['generated_response']], answers)
    # mauve_scores = calculate_mauve([result['generated_response']], answers)

    # Compute the gaussian weighted mean of the scores
    r1f1_scores_dict[f'r1f1_{answer_type}'] = np.mean(r1f1_scores)
    r2f1_scores_dict[f'r2f1_{answer_type}'] = np.mean(r2f1_scores)
    rLf1_scores_dict[f'rLf1_{answer_type}'] = np.mean(rLf1_scores)
    similarities_dict[f'similarity_{answer_type}'] = np.mean(similarities)
    # mauve_scores_dict[f'mauve_{answer_type}'] = np.mean(mauve_scores)

    metrics.update(r1f1_scores_dict)
    metrics.update(r2f1_scores_dict)
    metrics.update(rLf1_scores_dict)
    metrics.update(similarities_dict)
    # metrics.update(mauve_scores_dict)
    return metrics

def calculate_f1_score(generated_answers, gold_answers, n='1'):
    """
    Calculates the F1 score based on the overlap between two lists of strings.

    Args:
        generated_answers (list[str]): Generated answers.
        gold_answers (list[str]): Gold standard answers.
        n (string): N-gram length for ROUGE-N calculation. Default is 1 (ROUGE-1).

    Returns:
        float: F1 score based on the overlap of the answers.
    """
    scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)

    # Calculate f1 score for each pair
    scores = []
    for generated in generated_answers:
        for gold in gold_answers:
            score = scorer.score(gold, generated)
            rouge_n_f1 = score[f'rouge{n}'].fmeasure
            scores.append(rouge_n_f1)

    return scores

def calculate_cosine_similarity(embedding_model, generated_answers, gold_answers):
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
            similarity = cosine_similarity([gen_emb], [gold_emb])[0][0]
            similarities.append(similarity)

    return similarities


# def calculate_mauve(generated_answers, gold_answers):
#     """
#     Calculates Mauve score between the generated and gold answers.

#     Args:
#         generated_answers (list[str]): Generated answers.
#         gold_answers (list[str]): Gold standard answers.

#     Returns:
#         float: Mauve score between the answers.
#     """
#         # Generate embeddings

#     # Calculate cosine similarity for each pair
#     similarities = []
#     for gen_answer in generated_answers:
#         if gen_answer:
#             mauve_scores = []
#             gold_answers = [gold_answer for gold_answer in gold_answers if gold_answer]
#             gen_answers = [gen_answer]*len(gold_answers)
#             print(gen_answers)
#             print(gold_answers)
#             score = mauve.compute_mauve(p_text=gen_answers, q_text=gold_answers, device_id=0, max_text_length=256, verbose=False)    
#             mauve_scores.append(score.mauve)
#             print(score.mauve)
#             similarities.append(np.max(mauve_scores))

#     return similarities

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
    
if __name__ == "__main__":
    run = '19'
    time = '09-10_12-09'

    dir = f'./results_mmlu/run{run}_{time}/'
    all_entries = os.listdir(dir)
    files_ = [entry for entry in all_entries if os.path.isfile(os.path.join(dir, entry))]
    names = [filename.replace('evaluation_', '').replace('.pkl', '') for filename in files_ if filename.startswith('evaluation')]

    eval_dfs = [pd.read_pickle(f'{dir}/evaluation_{name}.pkl') for name in names]
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # metrics = ['r1f1', 'r2f1', 'rLf1', 'similarity', 'mauve']
    metrics = ['r1f1', 'r2f1', 'rLf1', 'similarity']
    category = 'correct'
    # Initialize the dictionary to store computed means
    computed_means = {name: {metric: {} for metric in metrics} for name in names}
    
    for name, df in tzip(names, eval_dfs):
        print(name)
        result_df = compute_evaluation_metrics(embedding_model, df)
        print(result_df.columns)
        for metric in metrics:
            col = f'{metric}_{category}'
            computed_means[name][metric] = result_df[col].mean()
            
        print(computed_means[name])
    computed_means = convert_numpy_types(computed_means)

    with open(f"{dir}eval_results.json", "w") as outfile: 
        json.dump(computed_means, outfile)    
    