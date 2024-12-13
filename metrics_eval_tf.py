import pandas as pd
import numpy as np

from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import mauve

import os
import json
from tqdm.contrib import tzip
import ast
    
def compute_evaluation_metrics(embedding_model, test_data, answer_type):
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
        result.update(calculate_metrics(embedding_model, result, answer_type))

        data_and_evaluation.append(result)

    return pd.DataFrame(data_and_evaluation)

def check_type(variable):
    if isinstance(variable, str):
        return variable
    else:
        print("This is a sequence (array-like).")
        print(variable[0])
        return check_type(variable[0])

def calculate_metrics(embedding_model, result, answer_type):
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

    if answer_type == 'correct':
        answers = result['correct_answers']
    else:
        answers = result[f'{answer_type}_answers']
    # answers = [answers[0]]
    generated_response = [str(result['generated_response'])]*len(answers)
    answers = [str(answer) for answer in answers]
    # generated_response = [str(result['best_answer'])]*len(answers)
    # Calculate F1 scores and similarities
    r1f1_scores = calculate_f1_score(generated_response, answers, n='1')
    r2f1_scores = calculate_f1_score(generated_response, answers, n='2')
    rLf1_scores = calculate_f1_score(generated_response, answers, n='L')
    similarities = calculate_cosine_similarity(embedding_model, generated_response, answers)

    # Compute the gaussian weighted mean of the scores
    r1f1_scores_dict[f'r1f1'] = np.mean(r1f1_scores)
    r2f1_scores_dict[f'r2f1'] = np.mean(r2f1_scores)
    rLf1_scores_dict[f'rLf1'] = np.mean(rLf1_scores)
    similarities_dict[f'similarity'] = np.mean(similarities)

    metrics.update(r1f1_scores_dict)
    metrics.update(r2f1_scores_dict)
    metrics.update(rLf1_scores_dict)
    metrics.update(similarities_dict)
    
    return metrics

def compute_evaluation_mauve(embedding_model, test_data, answer_type):
    """
    Formats and combines evaluation results into a single DataFrame.

    Args:
        test_data (DataFrame): Original test data.

    Returns:
        DataFrame: Combined evaluation results.
    """
    answers  = []
    generated_answers = []
    for i in range(len(test_data)):
        answer_name = answer_type+"_answers"
        answer = test_data.loc[i, answer_name]
        # answer = [test_data.loc[i, answer_name][0]]
        print(answer)
        answers += answer
        # generated_answers += [str(test_data.loc[i,'generated_response'])]*len(answer)
        generated_answers += [str(test_data.loc[i,'best_answer'])]*len(answer)
    
    print(len(generated_answers), len(answers))
    mauve_score = calculate_mauve(embedding_model, generated_answers, answers)

    return mauve_score

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
    # print(gold_answers)
    generated_embeddings = embedding_model.encode(generated_answers)
    gold_embeddings = embedding_model.encode(gold_answers)

    # Calculate cosine similarity for each pair
    similarities = []
    # for gen_emb in generated_embeddings:
    #     for gold_emb in gold_embeddings:
    # print(generated_embeddings.shape, gold_embeddings.shape)
    similarity = cosine_similarity(generated_embeddings, gold_embeddings)[0][0]
    # print(similarity)
    similarities.append(similarity)

    return similarities


def calculate_mauve(embedding_model, generated_answers, answers):
    """
    Calculates Mauve score between the generated and gold answers.

    Args:
        generated_answers (list[str]): Generated answers.
        gold_answers (list[str]): Gold standard answers.

    Returns:
        float: Mauve score between the answers.
    """
        # Generate embeddings

    similarities = []

    p_features = embedding_model.encode(generated_answers)
    q_features = embedding_model.encode(answers)
    score = mauve.compute_mauve(p_features=p_features, q_features=q_features)
    # print(score.mauve)
    similarities.append(score.mauve)

    return similarities

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

    dir = './results/run30_12-10_13-07'
    
    data_names = ['Token25', 'Token50', 'Token100']
    for data_name in data_names:
        save_dir = f'{dir}/{data_name}'
        os.makedirs(save_dir, exist_ok=True)
        test_data = pd.read_pickle(f'{dir}/evaluation_2D_{data_name}.pkl')
        test_data.to_csv(f'{save_dir}/evaluation_2D_{data_name}.csv')
        print(test_data.columns)

        
        # test_data['correct_answers'] = test_data['correct_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
        # test_data['incorrect_answers'] = test_data['incorrect_answers'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [x])
        test_data = test_data[['question', 'generated_response', 'best_answer', 'correct_answers', 'incorrect_answers']]
        # test_data  = test_data.sample(n=50, random_state=42)
        test_data = test_data.reset_index(drop=True)
        print('length of test_data: ', len(test_data))
        test_data.to_csv(f'{save_dir}/evaluation_2D_{data_name}.csv')


        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_model = SentenceTransformer(embedding_model_name)
        
        metrics = ['r1f1', 'r2f1', 'rLf1', 'similarity']
        answer_types = ['correct', 'incorrect']
        # Initialize the dictionary to store computed means
        computed_means = {answer_type: {metric: {} for metric in metrics} for answer_type in answer_types} 
        
        for answer_type in answer_types:
            result_df = compute_evaluation_metrics(embedding_model, test_data, answer_type)
            print(result_df.columns)
            for metric in metrics:
                computed_means[answer_type][metric] = result_df[metric].mean()
                
            mauve_score = compute_evaluation_mauve(embedding_model, test_data, answer_type)
            computed_means[answer_type]['mauve'] = mauve_score[0]
            if answer_type == 'correct':
                test_data[['correct_r1f1', 'correct_r2f1', 'correct_rLf1', 'correct_similarity']] = result_df[['r1f1', 'r2f1', 'rLf1', 'similarity']]
            else:
                test_data[['incorrect_r1f1', 'incorrect_r2f1', 'incorrect_rLf1', 'incorrect_similarity']] = result_df[['r1f1', 'r2f1', 'rLf1', 'similarity']]
            
        computed_means = convert_numpy_types(computed_means)

        with open(f"{save_dir}/eval_results.json", "w") as outfile: 
            json.dump(computed_means, outfile)    
        test_data.to_csv(f"{save_dir}/result_df.csv")