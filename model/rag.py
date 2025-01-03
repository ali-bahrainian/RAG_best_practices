import numpy as np
import pandas as pd
import re
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import mauve
from tqdm import tqdm

class RAG:
    """
    Integrates a retriever and a language model to generate responses based on retrieved information.
    This class uses a retriever to fetch relevant document (chunks) based on a query and employs a language model
    to synthesize responses incorporating the information from these documents and the original query.
    Its main functionality is exposed throug the evaluate method, which assesses the system's performance using test data.

    Attributes:
        retriever (Retriever): Instance of Retriever class for fetching documents.
        language_model (LanguageModel): Instance of LanguageModel class for generating text.
        system_prompt (str): A predefined prompt to prepend to each input.
    """

    def __init__(self, retriever, language_model, system_prompt, repeat_system_prompt, top_k_docs, expand_query, top_k_titles, stride, query_len, do_sample, temperature, top_p, num_beams, max_new_tokens, batch_size, kb_10K, icl_kb, icl_kb_incorrect, focus):
        self.retriever = retriever
        self.language_model = language_model

        self.system_prompt = system_prompt
        self.repeat_system_prompt = repeat_system_prompt

        self.retrieval_kwargs = {
            "k": top_k_docs,
            "expand_query": expand_query,
            "k_titles": top_k_titles,
            "focus": focus
        }

        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.stride = stride
        self.query_len = query_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.Kb_10K = kb_10K
        self.icl_kb = icl_kb
        self.icl_kb_incorrect = icl_kb_incorrect
        self.focus= focus

    def _prompt_template(self, query, docs_text, docs_correct_answer, docs_incorrect_answer):
        if docs_text:
            # Format the prompt based on the type of knowledge base
            if not self.icl_kb:
                system_prompt = self.system_prompt + " considering these information\n" if self.system_prompt else ""
                repeat_prompt = self.system_prompt + "\n" if self.repeat_system_prompt else ""
                docs_str = "\n".join("- " + re.sub(r'[\t\n\r\f\v]', ' ', doc) for doc in docs_text) + "\n---\n"
                rag_prompt =  f"{system_prompt}{docs_str}{repeat_prompt}Question:{query} Answer:"
            else:
                system_prompt = self.system_prompt + " considering these examples\n" if self.system_prompt else ""
                repeat_prompt = self.system_prompt + "\n" if self.repeat_system_prompt else ""
                if self.icl_kb_incorrect:
                    docs_str = "\n".join("- Question:" + question + ", Correct Answer:" + str(correct) + "\n- Question:" + question + ", Incorrect Answer:" + str(incorrect) for question, correct, incorrect  in zip(docs_text, docs_correct_answer, docs_incorrect_answer)) + "\n---\n"
                else:
                    docs_str = "\n".join("- Question:" + question + ", Correct Answer:" + str(correct) for question, correct  in zip(docs_text, docs_correct_answer)) + "\n---\n"
                rag_prompt =  f"{system_prompt}{docs_str}{repeat_prompt}Question:{query}, Correct Answer:"
        else:
            system_prompt = self.system_prompt + "\n" if self.system_prompt else ""
            rag_prompt = f"{system_prompt}Question:{query} Answer:"
        return self.language_model.instruct_start + rag_prompt + self.language_model.instruct_end


    def evaluate(self, test_data):
        """
        Evaluates the RAG instance using the provided test data. It processes the data in batches, computing metrics

        Args:
            test_data (DataFrame): DataFrame containing query and expected response pairs.
            batch_size (int): Size of the batch for processing.

        Returns:
            DataFrame: DataFrame containing test data, generated responses, and evaluation scores.
        """
        self.test_data = test_data

        # Prepare batches
        test_samples_generation = [{'query_id': idx, 'query': query} for idx, query in test_data['question'].items()]
        test_batches_generation = [test_samples_generation[i:i + self.batch_size] for i in range(0, len(test_samples_generation), self.batch_size)]

        # Evaluate generation
        results_generation = self._evaluate_generation(test_batches_generation)
        results_df = self._compute_evaluation_metrics(test_data, results_generation)
        mauve_score = self._compute_evaluation_mauve(self.retriever.embedding_model, results_df)
        # Combine and format results
        return results_df, mauve_score


    def _evaluate_generation(self, test_batches):
        """
        Generates responses for each batch in the test data.

        Args:
            test_batches (List[List[Dict]]): List of test batches.

        Returns:
            Dict: Generated responses for each test entry.
        """
        results_gen = {}
        for batch in tqdm(test_batches, desc="Calculating Generation"):
            batch_queries = [item['query'] for item in batch]
            input_texts, generated_responses = self._generate(batch_queries)

            for i, item in enumerate(batch):
                query_id = item['query_id']
                results_gen[query_id] = {'input_text': input_texts[i], 'generated_response': generated_responses[i]}
        return results_gen


    def _generate(self, query_batch):
        """
        Generates responses to queries by first retrieving relevant information and then synthesizing answers.

        Args:
            query_batch (list[str]): The input queries.
            top_k_docs (int): Number of top relevant documents to retrieve.
            max_new_tokens (int): Maximum token length of the response.
            query_length is measured in chars

        Returns:
            list[str]: Generated responses.
        """
        stride = self.stride if self.stride > 0 else self.max_new_tokens

        retrieval_kwargs = self.retrieval_kwargs
        if self.icl_kb:
            retrieval_kwargs['icl_kb_idx_batch'] = [self._query_idx(query) for query in query_batch]
        # Retrieve documents
        docs_batch = self.retriever.retrieve(query_batch, **retrieval_kwargs)
        context_batch = self._format_context(query_batch, docs_batch)

        responses_enc = [[] for _ in query_batch]

        done_batch = [False for _ in query_batch]
        # Generate responses based on the documents retrieved from each stride 
        for i in range(0, self.max_new_tokens, stride):
            # Expand the query with the generated response from the previous stride
            query_reponse_batch = [(q+" "+self.language_model.tokenizer.decode(r))[-self.query_len:] for q, r in zip(query_batch, responses_enc)]
            # Retrieve documents based on the expanded query
            docs_batch = self.retriever.retrieve(query_reponse_batch, **retrieval_kwargs)
            # Format context
            _context_batch = self._format_context(query_batch, docs_batch)
            running_context_batch = [c+self.language_model.tokenizer.decode(r) for c, r in zip(_context_batch, responses_enc)]
            # Generate responses
            new_responses_str, _done_batch = self.language_model.generate(running_context_batch, self.do_sample, self.temperature, self.top_p, self.num_beams, max_new_tokens=stride)
           
            done_batch = [bool(d + _d) for d, _d in zip(done_batch, _done_batch)]

            for idx, (done, new_response_str, running_context, response_enc) in enumerate(zip(done_batch, new_responses_str, running_context_batch, responses_enc)):
                if not done:
                    new_response_str = new_response_str[len(running_context):]
                    new_response_enc = self.language_model.tokenizer.encode(new_response_str, add_special_tokens=False)
                    # Append the new response to the previous response
                    responses_enc[idx] = response_enc + new_response_enc


        responses_str = [self.language_model.tokenizer.decode(r, add_special_tokens=False) for r in responses_enc]
        return context_batch, responses_str

    def _format_context(self, queries, retrieved_docs):
        """
        Formats the input for the language model by combining retrieved docuemnts with queries.

        Args:
            queries (list[str]): A list of query dictionaries.
            retrieved_docs (list[list[str]]): A list containing lists of retrieved docs for each query.

        Returns:
            list[str]: Formatted input texts for the language model.
        """
        input_texts = []
        for docs, query in zip(retrieved_docs, queries):
            docs_text = [doc['text'] for doc in docs]
            docs_correct_answer, docs_incorrect_answer = None, None
            if self.icl_kb:
                docs_correct_answer = [doc['correct_answer'] for doc in docs]
                docs_incorrect_answer = [doc['incorrect_answer'] for doc in docs]
            formatted_input = self._prompt_template(query, docs_text, docs_correct_answer, docs_incorrect_answer)
            input_texts.append(formatted_input)
        return input_texts

    
    def _compute_evaluation_metrics(self, test_data, results_gen):
        """
        Formats and combines evaluation results into a single DataFrame.

        Args:
            test_data (DataFrame): Original test data.
            results_gen (Dict): Generated response results.

        Returns:
            DataFrame: Combined evaluation results.
        """
        data_and_evaluation = []
        for qid in test_data.index:
            result = test_data.loc[qid].to_dict()
            result.update({
                'input_text': results_gen[qid]['input_text'],
                'generated_response': results_gen[qid]['generated_response']
            })
            result.update(self._calculate_metrics(result))

            data_and_evaluation.append(result)

        return pd.DataFrame(data_and_evaluation)

    def _calculate_metrics(self, result):
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

        answers = result['correct_answers'] + result['best_answer']*2
        answers = [answer for answer in answers if answer]

        # Calculate F1 scores and similarities
        r1f1_scores = self._calculate_f1_score([result['generated_response']], answers, n='1')
        r2f1_scores = self._calculate_f1_score([result['generated_response']], answers, n='2')
        rLf1_scores = self._calculate_f1_score([result['generated_response']], answers, n='L')
        similarities = self._calculate_cosine_similarity([result['generated_response']], answers)

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

    def _calculate_f1_score(self, generated_answers, reference_answers, n='1'):
        """
        Calculates the F1 score based on the overlap between two lists of strings.

        Args:
            generated_answers (list[str]): Generated answers.
            reference_answers (list[str]): Reference answers.
            n (string): N-gram length for ROUGE-N calculation. Default is 1 (ROUGE-1).

        Returns:
            float: F1 score based on the overlap of the answers.
        """
        scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)

        # Calculate f1 score for each pair
        scores = []
        for generated in generated_answers:
            for ref in reference_answers:
                score = scorer.score(ref, generated)
                rouge_n_f1 = score[f'rouge{n}'].fmeasure
                scores.append(rouge_n_f1)

        return scores

    def _calculate_cosine_similarity(self, generated_answers, reference_answers):
        """
        Calculates cosine similarity between the generated and gold answers.

        Args:
            generated_answers (list[str]): Generated answers.
            reference_answers (list[str]): Gold standard answers.

        Returns:
            float: Cosine similarity between the answers.
        """
         # Generate embeddings
        generated_embeddings = self.retriever.embedding_model.encode(generated_answers)
        ref_embeddings = self.retriever.embedding_model.encode(reference_answers)

        # Calculate cosine similarity for each pair
        similarities = []
        for gen_emb in generated_embeddings:
            for ref_emb in ref_embeddings:
                similarity = cosine_similarity([gen_emb], [ref_emb])[0][0]
                similarities.append(similarity)

        return similarities

    def _compute_evaluation_mauve(self, embedding_model, results_df):
        """
        Formats and combines evaluation results into a single DataFrame.

        Args:
            test_data (DataFrame): Original test data.

        Returns:
            DataFrame: Combined evaluation results.
        """
        answers  = []
        generated_answers = []
        for i in range(len(results_df)):
            answer = results_df.loc[i, "correct_answers"]
            answers += answer + results_df.loc[i, 'best_answer']*2
            answers = [answer for answer in answers if answer]
            if len(answers) == 0:
                print(results_df.loc[i])
            generated_answers += [str(results_df.loc[i,'generated_response'])]*len(answers)
        
        mauve_scores = self._calculate_mauve(embedding_model, generated_answers, answers)

        return mauve_scores
    
    def _calculate_mauve(self, embedding_model, generated_answers, reference_answers):
        """
        Calculates Mauve score between the generated and gold answers.

        Args:
            generated_answers (list[str]): Generated answers.
            reference_answers (list[str]): Gold standard answers.

        Returns:
            float: Mauve score between the answers.
        """
        # Generate embeddings
        p_features = embedding_model.encode(generated_answers)
        q_features = embedding_model.encode(reference_answers)
        score = mauve.compute_mauve(p_features=p_features, q_features=q_features)

        return score.mauve

    def _query_idx(self, query):
        return self.test_data.index[self.test_data['question'] == query].tolist()[0]