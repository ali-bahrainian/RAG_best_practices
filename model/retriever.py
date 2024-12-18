from faiss import IDSelectorArray, SearchParameters
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

import spacy
import faiss

# Load the English model
nlp = spacy.load("en_core_web_sm")

class Retriever:
    """
    Handles the retrieval of relevant documents from a pre-built FAISS index.
    Enables querying with sentence transformers embeddings.

    Attributes:
        index (faiss.Index): FAISS index for fast similarity search.
        doc_info (pd.DataFrame): DataFrame containing detailed information about documents.
        documents (list of str): List of original documents.
        embedding_model (SentenceTransformer): Model used for embedding the documents and queries.
    """

    def __init__(self, index, doc_info, embedding_model_name, model_loader_seq2seq, index_titles):
        """Initializes the Retriever class with necessary components.

        Args:
            index: FAISS index for fast retrieval.
            doc_info (DataFrame): DataFrame containing info about embedded document; aligned indices with index embeddings.
            documents (list): List of original documents.
            embedding_model_name (str): Name of the sentence transformer model.
        """
        self.index = index
        self.doc_info = doc_info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)
        self.sent_info = None
        self.index_sents = None

        self.model_seq2seq = model_loader_seq2seq.model
        self.tokenizer_seq2seq = model_loader_seq2seq.tokenizer
        # Define text-query pairs for query expansion
        self.text_query_pairs = [
            {"text": "Mitochondria play a crucial role in cellular respiration and energy production within human cells.", "query": "Cell Biology, Mitochondria, Energy Metabolism"},
            {"text": "The Treaty of Versailles had significant repercussions that contributed to the onset of World War II.", "query": "World History, Treaty of Versailles, World War II"},
            {"text": "What are the implications of the Higgs boson discovery for particle physics and the Standard Model?", "query": "Particle Physics, Higgs Boson, Standard Model"},
            {"text": "How did the Silk Road influence cultural and economic interactions during the Middle Ages?", "query": "Silk Road, Middle Ages, Cultural Exchange"}
        ]
        self.index_titles = index_titles

    def build_index(self, documents):
        """
        Builds a FAISS index from document embeddings for efficient similarity searches which
        includes embedding document chunks and initializing a FAISS index with these embeddings.

        Args:
            chunk_size (int): The size of each text chunk in tokens.
            overlap (int): The number of tokens that overlap between consecutive chunks.

        Returns:
            faiss.IndexFlatIP: The FAISS index containing the embeddings of the document chunks.
        """
        embeddings = self.embed_sents(documents)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return index

    def embed_sents(self, documents):
        """
        Generates embeddings for document chunks.

        The process involves:
        1. Preparing chunks of documents:
          - Splits each document into overlapping chunks based on `chunk_size` and `overlap`.
        2. Encoding these chunks/documents into embeddings using the Sentence Transformer.

        Args:
            chunk_size (int): Size of each chunk in tokens.
            overlap (int): Overlap between consecutive chunks in tokens.

        Returns:
            np.ndarray: An array of embeddings for all the documents (chunks).
        """
        self.sent_info = self.prepare_sents(documents)
        self.sent_info = pd.DataFrame(self.sent_info)
        embeddings = self.embedding_model.encode(self.sent_info["text"].tolist(), show_progress_bar=True)
        self.sent_info['embedding'] = embeddings.tolist()

        return np.array(embeddings)
    
    def prepare_sents(self, documents):
        """
        Splits each document into sentences and
        creates dictionary for DataFrame associated with index.

        Returns:
            Tuple[List[str], List[dict]]: Tuple containing list of all sents and their info.
        """
        sent_info = []
        sent_id = 0
        for document in documents:
            
            doc = nlp(document)
            sents = [sent.text for sent in doc.sents]
            
            # Prepend same document to its chunks and store document/chunk details
            for sent in sents:
                sent_dict = {"text": sent, "org_sent_id": sent_id}
                sent_info.append(sent_dict)
                sent_id += 1
        return sent_info

    def retrieve(self, query_batch, k, expand_query, k_titles, icl_kb_idx_batch=None, focus=None):
        """
        Retrieves the top-k most similar documents for each query in a batch of queries.

        Args:
            query_batch (list of str): List of query strings.
            k (int): Number of documents to retrieve.

        Returns:
            List[List[dict]]: List of lists containing formatted results of retrieved documents for each query.
        """

        if k == 0:
            return [[] for _ in query_batch]

        if expand_query:
            # Expand the query using a seq2seq model
            eq_prompt_batch_str = []
            for query in query_batch:
                examples = self.text_query_pairs.copy()
                examples.append({"text": query, "query": ""})
                eq_prompt = "\n".join([f"Question: {example['text']}\nQuery Keywords: {example['query']}" for example in examples])
                eq_prompt_batch_str.append(eq_prompt)

            eq_prompt_batch_enc = self.tokenizer_seq2seq(eq_prompt_batch_str, return_tensors='pt', padding=True).to(self.device)
            eq_batch_enc = self.model_seq2seq.generate(**eq_prompt_batch_enc, max_length=25, num_return_sequences=1)
            eq_batch = self.tokenizer_seq2seq.batch_decode(eq_batch_enc, skip_special_tokens=True)
            eq_batch = [eq.split(", ") for eq in eq_batch] # Split the expanded queries

            # Encode the expanded queries and search the index for similar titles
            eq_batch_indexed = [(eq, i) for i, eqs in enumerate(eq_batch) for eq in eqs]
            eq_batch_flat = [eq for eq, _ in eq_batch_indexed]
            eq_embeddings = self.embedding_model.encode(eq_batch_flat, show_progress_bar=False)
            _, indices_eq = self.index_titles.search(np.array(eq_embeddings), k_titles)

            # Retrieve the indices of the documents associated with the similar titles
            indices_eq_batch = [[] for _ in range(len(query_batch))]
            for ids, (_, i) in zip(indices_eq, eq_batch_indexed):
                indices_eq_batch[i].append(self.doc_info[self.doc_info['org_doc_id'].isin(ids)].index.tolist())
        else:
            # If not expanding the query, set the indices to an empty list
            if icl_kb_idx_batch:
                # Remove the correct answer from the retrieved documents
                all_ids_batch = [list(range(self.index.ntotal)) for _ in range(len(query_batch))]
                for all_ids, icl_kb_idx in zip(all_ids_batch, icl_kb_idx_batch):
                    all_ids.remove(icl_kb_idx)
                all_ids_batch = [[all_ids] for all_ids in all_ids_batch]
                indices_eq_batch = all_ids_batch
            else:
                indices_eq_batch = [[] for _ in range(len(query_batch))]

        # Batch encode the queries
        query_embeddings = self.embedding_model.encode(query_batch, show_progress_bar=False)

        # Process each query separately
        results_batch = []
        for query_embedding, ids_filter in zip(query_embeddings, indices_eq_batch):
            ids_filter = ids_filter if ids_filter else [list(range(self.index.ntotal))]

            id_filter_set = set()
            for id_filter in ids_filter:
                id_filter_set.update(id_filter)

            id_filter = list(id_filter_set)
            id_selector = IDSelectorArray(id_filter)
            # Search the index for similar documents, retrieve a larger set of documents
            similarities, indices = self.index.search(np.array([query_embedding]), k, params=SearchParameters(sel=id_selector))
            indices, similarities = indices[0], similarities[0]
            
            # Focus on the most relevant sentences from the retrieved documents
            if focus:
                docs = self.doc_info.loc[indices]["text"].tolist()
                self.index_sents = self.build_index(docs)   
                similarities, indices = self.index_sents.search(np.array([query_embedding]), focus)
                indices, similarities = indices[0], similarities[0]

            icl_kb = icl_kb_idx_batch!=None
            if focus:
                # Retrieve the most relevant sentences from the retrieved documents
                results_batch.append([self._create_result(idx, sim, icl_kb, focus) for idx, sim in zip(indices[:focus], similarities)])
            else:
                results_batch.append([self._create_result(idx, sim, icl_kb, focus) for idx, sim in zip(indices[:k], similarities)])

        return results_batch


    def _create_result(self, idx, score, icl_kb, focus):
        """
        Creates/builds a result dictionary of the retrieved document.

        Args:
            idx (int): Index of the result/document in doc_info.
            score (float): Similarity (& Diversity) score of document.

        Returns:
            dict: Dictionary containing the document text and additional information.
        """
        if focus: 
            # Retrieve the most relevant sentences from the retrieved documents
            sent = self.sent_info.iloc[idx]
            result_dict = {
            "text": sent["text"],
            "sent_id": sent["org_sent_id"],
            "score": score
        }
        else:
            doc = self.doc_info.iloc[idx]
            # Create the result dictionary
            result_dict = {
                "text": doc["text"],
                "doc_id": doc["org_doc_id"],
                "score": score
            }

            if icl_kb:
                # Include the correct and incorrect answers for ICL KB
                result_dict['correct_answer'] = doc["correct_answer"]
                result_dict['incorrect_answer'] = doc["incorrect_answer"]

        return result_dict