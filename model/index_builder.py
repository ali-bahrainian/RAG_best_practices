import faiss
from gensim.corpora import Dictionary
from gensim.models import LdaModel
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
        metadata (list of dict, optional): Metadata associated with each document.
        # TODO

    Attributes:
        documents (list of str): Original list of documents.
        titles (list of str): Original list of titles.
        embedding_model (SentenceTransformer): Sentence transformer model used for embedding documents.
        metadata (list of dict, optional): Metadata associated with each document.
        index (faiss.Index): FAISS index for efficiently searching documents based on their embeddings.
        doc_info (pd.DataFrame): DataFrame containing information about documents, including text, embedding, and topic distribution.
        lda_model (gensim.models.ldamodel.LdaModel): LDA model for identifying topics in documents.
        dictionary (gensim.corpora.Dictionary): Gensim dictionary for converting text to vectors.
        corpus (list of gensim.matutils.SparseVector): Gensim corpus representing documents as bag-of-words vectors.
    """

    def __init__(self, documents_df, embedding_model_name, expand_query, metadata_cols, tokenizer_model_name, chunk_size, overlap, num_topics, passes, icl_kb, multi_lingo, mmlu_kb):
        """
        Initializes the IndexBuilder class with necessary components.
        """
        self.expand_query = expand_query
        self.icl_kb = icl_kb
        self.multi_lingo = multi_lingo
        self.mmlu_kb = mmlu_kb

        if not self.icl_kb:
            if not self.mmlu_kb:
                self.documents = documents_df['text_en'].tolist()
                self.documents_de = documents_df['text_de'].tolist()
                self.documents_fr = documents_df['text_fr'].tolist()
                self.titles = documents_df['title_en'].tolist()
                self.metadata = documents_df[metadata_cols].to_dict(orient='records') if metadata_cols else None
            else:
                self.documents = documents_df['text_en'].tolist()
                self.metadata = None
        else:
            self.documents = documents_df['question'].tolist()
            self.titles = None
            self.metadata = None
            self.best_answers = documents_df['best_answer'].tolist()
            self.incorrect_answers = documents_df['incorrect_answers'].tolist()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)

        if tokenizer_model_name:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer_model_name)
            self.tokenizer.model_max_length = 100_000
        else:
            self.tokenizer = None

        self.chunk_size, self.overlap, self.num_topics, self.passes = chunk_size, overlap, num_topics, passes

    def initialize_components(self):
        """
        The primary method to be used externally for activating the index building mechanism.
        It orchestrates the creation of a searchable index and the associated DataFrame containing
        embeddings, text, metadata, and topic distributions.

        It follows these key steps:
        1. Build an FAISS index using the document embeddings, for efficient similarity search.
        2. Train an LDA model to capture topic distributions within the documents and
           update the chunk information with the identified topic distributions

        The resulting index and DataFrame are aligned such that each embedding in the index corresponds
        to the associated text, metadata, and topic distribution.

        Args:
            chunk_size (int): Size of each chunk in tokens.
            overlap (int): Overlap between consecutive chunks in tokens.
            num_topics (int): Number of topics for the LDA model.
            passes (int): Number of training passes over the corpus.

        Returns:
            tuple: A tuple containing the FAISS index and document information as a DataFrame.
        """
        self._check_chunk_size() # Ensure chunk size is appropriate for model's input capacity

        self.doc_info = None
        self.index = self._build_index()
        self.title_index = None   # self._build_title_index() if not self.icl_kb else None  # None
        if self.expand_query:
            self.title_index = self._build_title_index() if not self.icl_kb else None  # None
        if not self.icl_kb:
            self._train_lda_model(self.num_topics, self.passes)
            self._update_doc_with_topics()

        return self.index, self.title_index, self.doc_info

    def _check_chunk_size(self, tolerance_ratio=0.1):
        """
        Validates if the chunk size is within the model's maximum input size, considering metadata.

        Args:
            chunk_size (int): Size of each chunk in tokens.
            tolerance_ratio (float): Ratio of (chunk + metadata) length to max input size.
        """
        # Estimated length considering metadata
        total_length = self.chunk_size + (len(self._format_metadata(self.metadata[0])) if self.metadata else 0)

        if total_length > self.embedding_model.get_max_seq_length() * (1 - tolerance_ratio):
            raise ValueError(f"Combined length of chunk and metadata exceeds the allowed maximum.")

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
          - If available, metadata for each document is prepended to each chunk.
        2. Encoding these chunks/documents into embeddings using the Sentence Transformer.

        Args:
            chunk_size (int): Size of each chunk in tokens.
            overlap (int): Overlap between consecutive chunks in tokens.

        Returns:
            np.ndarray: An array of embeddings for all the documents (chunks).
        """
        doc_info = self._prepare_docs()
        self.doc_info = pd.DataFrame(doc_info)
        embeddings = self.embedding_model.encode(self.doc_info['doc'].tolist(), show_progress_bar=True)
        self.doc_info['embedding'] = embeddings.tolist()

        return np.array(embeddings)

    def _prepare_docs(self):
        """
        Splits each document into overlapping chunks, appending metadata and
        creates dictionary for DataFrame associated with index.

        Args:
            chunk_size (int): Size of each chunk in tokens.
            overlap (int): Overlap between consecutive chunks in tokens.

        Returns:
            Tuple[List[str], List[dict]]: Tuple containing list of all docs and their info.
        """
        doc_info = []

        for org_doc_id in range(len(self.documents)):
            if self.mmlu_kb:
                doc_en = self.documents[org_doc_id]
            else:  
                doc_en = self.documents[org_doc_id]
                doc_fr = self.documents_fr[org_doc_id]
                doc_de = self.documents_de[org_doc_id]
            if not self.multi_lingo:
                doc = doc_en
            else:
                doc = np.random.choice([doc_en, doc_fr, doc_de])

            if doc is None:
                doc = doc_en
            
            # Breaks document into chunks but we will still call them documents
            docs = self._create_chunks(doc)
            metadata = self.metadata[org_doc_id] if self.metadata else None

            # Prepend same document metadata to its chunks and store document/chunk details
            for doc in docs:
                text = f"{self._format_metadata(metadata)}{doc}" if metadata else doc
                doc_dict = {"text": text, "metadata": metadata, "org_doc_id": org_doc_id, "doc": doc}
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

    def _format_metadata(self, metadata):
        """
        Formats metadata for inclusion in docs/chunks.

        Args:
            metadata (dict): Metadata for a document.

        Returns:
            str: Formatted metadata as a string.
        """
        if not metadata:
            return ""
        return ", ".join([value if value is not None else "" for value in metadata.values()]) + ": "

    def _train_lda_model(self, num_topics, passes):
        """
        Trains an LDA model on the corpus to uncover latent topics within the documents

        Args:
            num_topics (int): Number of topics for the LDA model.
            passes (int): Number of training passes over the corpus.
        """
        self._build_corpus()
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, passes=passes)

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

    def _update_doc_with_topics(self):
        """
        Updates document (chunks) info with topic distributions.
        """
        topic_distributions = []
        for _, doc in self.doc_info.iterrows():
            bow = self.dictionary.doc2bow(self._preprocess_text(doc['doc']))
            topic_distribution = self._get_topic_distribution(bow)
            topic_distributions.append(topic_distribution)

        self.doc_info['topic_distribution'] = topic_distributions

    def _get_topic_distribution(self, doc_bow):
        """
        Gets the topic distribution for a document.

        Args:
            doc_bow: Bag-of-words representation of a documene.

        Returns:
            List[float]: Normalized topic distribution for the documene.
        """
        return [prob for _, prob in self.lda_model.get_document_topics(doc_bow, minimum_probability=1e-3)]