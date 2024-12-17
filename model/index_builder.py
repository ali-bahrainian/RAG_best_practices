import faiss
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer


class IndexBuilder:
    """
    Builds an index from a collection of documents.
    Its main functionality is exposed through the `initialize_components` method, which activates the index building and topic modeling.

    Args:
        documents (list of str): List of documents to index.
        embedding_model_name (str): Name of the sentence transformer model to use for document embedding.
        # TODO

    Attributes:
        documents (list of str): Original list of documents.
        titles (list of str): Original list of titles.
        embedding_model (SentenceTransformer): Sentence transformer model used for embedding documents.
        index (faiss.Index): FAISS index for efficiently searching documents based on their embeddings.
        doc_info (pd.DataFrame): DataFrame containing information about documents, including text and embeddings.
        corpus (list of gensim.matutils.SparseVector): Gensim corpus representing documents as bag-of-words vectors.
    """

    def __init__(self, documents_df, embedding_model_name, expand_query, tokenizer_model_name, chunk_size, overlap, passes, icl_kb, multi_lingo):
        """
        Initializes the IndexBuilder class with necessary components.
        """
        self.expand_query = expand_query
        self.icl_kb = icl_kb
        self.multi_lingo = multi_lingo

        if self.icl_kb:
            self.documents = documents_df['question'].tolist()
            self.titles = None
            self.best_answers = documents_df['best_answer'].tolist()
            self.incorrect_answers = documents_df['incorrect_answers'].tolist()
        else:
            self.documents = documents_df['text_en'].tolist()
            self.documents_de = documents_df['text_de'].tolist()
            self.documents_fr = documents_df['text_fr'].tolist()
            self.titles = documents_df['title_en'].tolist()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)

        if tokenizer_model_name:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_model_name)
            self.tokenizer.model_max_length = 100_000
        else:
            self.tokenizer = None

        self.chunk_size, self.overlap, self.passes = chunk_size, overlap, passes

    def initialize_components(self):
        """
        The primary method to be used externally for activating the index building mechanism.
        It orchestrates the creation of a searchable index and the associated DataFrame containing
        embeddings and text

        An FAISS index is build for the document embeddings, for efficient similarity search.

        The resulting index and DataFrame are aligned such that each embedding in the index corresponds
        to the associated text

        Args:
            chunk_size (int): Size of each chunk in tokens.
            overlap (int): Overlap between consecutive chunks in tokens.
            passes (int): Number of training passes over the corpus.

        Returns:
            tuple: A tuple containing the FAISS index and document information as a DataFrame.
        """
        self._check_chunk_size() # Ensure chunk size is appropriate for model's input capacity

        self.doc_info = None
        self.index = self._build_index()
        self.title_index = None   
        if self.expand_query:
            self.title_index = self._build_title_index() if not self.icl_kb else None  

        return self.index, self.title_index, self.doc_info

    def _check_chunk_size(self, tolerance_ratio=0.1):
        """
        Validates if the chunk size is within the model's maximum input size.

        Args:
            chunk_size (int): Size of each chunk in tokens.
            tolerance_ratio (float): Ratio of chunk length to max input size.
        """
        total_length = self.chunk_size 

        if total_length > self.embedding_model.get_max_seq_length() * (1 - tolerance_ratio):
            raise ValueError(f"Combined length of chunk exceeds the allowed maximum.")

    def _build_title_index(self):
        """
        Builds a FAISS index for the titles.

        Returns:
            faiss.IndexFlatIP: The FAISS index for the title embeddings.
        """
        title_embeddings = np.array(self.embedding_model.encode(self.titles, show_progress_bar=True))
        title_index = faiss.IndexFlatIP(title_embeddings.shape[1])
        title_index.add(title_embeddings)

        return title_index


    def _build_index(self):
        """
        Builds a FAISS index from document embeddings for efficient similarity searches which
        includes embedding document chunks and initializing a FAISS index with these embeddings.

        Args:
            chunk_size (int): The size of each text chunk in tokens.
            overlap (int): The number of tokens that overlap between consecutive chunks.

        Returns:
            faiss.IndexFlatIP: The FAISS index containing the embeddings of the document chunks.
        """
        embeddings = self._embed_documents()
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return index

    def _embed_documents(self):
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
        doc_info = self._prepare_docs()
        self.doc_info = pd.DataFrame(doc_info)
        embeddings = self.embedding_model.encode(self.doc_info['text'].tolist(), show_progress_bar=True)
        self.doc_info['embedding'] = embeddings.tolist()

        return np.array(embeddings)

    def _prepare_docs(self):
        """
        Splits each document into overlapping chunks and
        creates dictionary for DataFrame associated with index.

        Args:
            chunk_size (int): Size of each chunk in tokens.
            overlap (int): Overlap between consecutive chunks in tokens.

        Returns:
            Tuple[List[str], List[dict]]: Tuple containing list of all docs and their info.
        """
        doc_info = []

        for org_doc_id in range(len(self.documents)):

            if self.multi_lingo:
                doc_en = self.documents[org_doc_id]
                doc_fr = self.documents_fr[org_doc_id]
                doc_de = self.documents_de[org_doc_id]
                doc = np.random.choice([doc_en, doc_fr, doc_de])
                if not doc:
                   doc = doc_en 
            else:
                doc = self.documents[org_doc_id]
            
            # Breaks document into chunks but we will still call them documents
            docs = self._create_chunks(doc)

            # Prepend same document to its chunks and store document/chunk details
            for doc in docs:
                doc_dict = {"text": doc, "org_doc_id": org_doc_id}
                if self.icl_kb:
                    doc_dict['correct_answer'] = self.best_answers[org_doc_id]
                    doc_dict['incorrect_answer'] = self.incorrect_answers[org_doc_id][0]
                doc_info.append(doc_dict)

        return doc_info

    def _create_chunks(self, text):
        """
        Creates overlapping chunks from a text document using a tokenizer.

        Args:
            text (str): Document text.

        Returns:
            List[str]: List of text chunks.
        """
        
        tokens = text.split() if self.tokenizer is None else self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        # Iterate through the tokens and create overlapping chunks
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_str = " ".join(chunk_tokens) if self.tokenizer is None else self.tokenizer.decode(chunk_tokens)
            if self.tokenizer and self.overlap:
                chunk_str = " ".join(chunk_str.split(" ")[1:-1]) # remove first and last word since might be subword
            chunks.append(chunk_str)

        return chunks

    def _build_corpus(self):
        """
        Builds a Gensim dictionary and corpus from the documents.
        """
        processed_docs = [self._preprocess_text(doc) for doc in self.documents]
        self.dictionary = Dictionary(processed_docs)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]

    def _preprocess_text(self, text):
        """
        Tokenizes and removes stopwords from the text.

        Args:
            text (str): Text to preprocess.

        Returns:
            List[str]: List of tokens after preprocessing.
        """
        return [token for token in simple_preprocess(text) if token not in STOPWORDS]