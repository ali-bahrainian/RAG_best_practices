import torch
from torch.nn import functional as F

class LanguageModel:
    """
    Encapsulates a transformer-based language model for text generation. This class provides functionalities
    to generate text based on input prompts and to calculate perplexity for given texts.

    Attributes:
        model (transformers.PreTrainedModel): The transformer-based language model for text generation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
        pad_token_id (int): Token ID used for padding in the model.
        is_chat_model (bool): Flag indicating if the model is specifically a chat model.
        instruct_start (str):
        instruct_end (str):
    """

    def __init__(self, model_loader, is_chat_model, instruct_tokens):
        """
        Initializes the LanguageModel with a pre-trained model and tokenizer.

        Args:
            model_loader (ModelLoader): Instance of ModelLoader containing the pre-trained model and tokenizer.
            is_chat_model (bool): Indicates if the model is a chat model.
            instruct_tokens (tuple()):
        """
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        
        # Set pad token as EOS for consistent padding
        self.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Flag for model's chat-specific prompt adaptation
        assert not is_chat_model or is_chat_model and instruct_tokens
        self.is_chat_model = is_chat_model
        self.instruct_start = '' if not self.is_chat_model else instruct_tokens[0]
        self.instruct_end = '' if not self.is_chat_model else instruct_tokens[1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, context_batch_str, do_sample, temperature, top_p, num_beams, max_new_tokens):
        """
        Generates text based on provided input texts using the language model.

        Args:
            context_batch_str (list[str]): List of context texts to seed the generation.
            max_new_tokens (int): Maximum length of the generated text in tokens.
        Returns:
            List[str]: Generated texts corresponding to each input text.
        """
        context_batch_enc = self.tokenizer(context_batch_str, return_tensors='pt', padding=True).to(self.device)

        generate_args = {
            'input_ids': context_batch_enc['input_ids'],
            'attention_mask': context_batch_enc['attention_mask'],
            'max_new_tokens': max_new_tokens,
            'pad_token_id': self.pad_token_id,
        }

        if do_sample:
            generate_args['do_sample'] = do_sample
            generate_args['temperature'] = temperature
            generate_args['top_p'] = top_p
            generate_args['num_beams'] = num_beams

        response_batch_enc = self.model.generate(**generate_args)

        response_batch_str = [self.tokenizer.decode(response_enc, skip_special_tokens=True) for response_enc in response_batch_enc]
        for i in range(len(response_batch_str)):
            if response_batch_str[i].strip() == "":
                print(f"Empty response detected from response_batch_enc: {response_batch_enc[i]}")
                raise ValueError("Empty result detected in the response batch.")

        done_batch = [response_enc[-1].item() == self.tokenizer.eos_token_id for response_enc in response_batch_enc]
        return response_batch_str, done_batch
    
    def generate_length(self, context_batch_str, do_sample, temperature, top_p, num_beams, max_length):
        """
        Generates text based on provided input texts using the language model.

        Args:
            context_batch_str (list[str]): List of context texts to seed the generation.
            max_length (int): The maximum length the generated tokens can have.
        Returns:
            List[str]: Generated texts corresponding to each input text.
        """
        context_batch_enc = self.tokenizer(context_batch_str, return_tensors='pt', padding=True).to(self.device)

        generate_args = {
            'input_ids': context_batch_enc['input_ids'],
            'attention_mask': context_batch_enc['attention_mask'],
            'max_length': max_length,
            'pad_token_id': self.pad_token_id,
        }

        if do_sample:
            generate_args['do_sample'] = do_sample
            generate_args['temperature'] = temperature
            generate_args['top_p'] = top_p
            generate_args['num_beams'] = num_beams

        response_batch_enc = self.model.generate(**generate_args)

        response_batch_str = [self.tokenizer.decode(response_enc, skip_special_tokens=True) for response_enc in response_batch_enc]
        done_batch = [response_enc[-1].item() == self.tokenizer.eos_token_id for response_enc in response_batch_enc]
        return response_batch_str, done_batch

    def generate_greedy_ensemble(self, contexts_str, weights, max_new_tokens, batch_size):
        # TOOO docstring

        contexts_enc = self.tokenizer(contexts_str, return_tensors='pt', padding=True).to(self.device)

        # Convert weights to a tensor and normalize them using softmax
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        weights = F.softmax(weights, dim=0)

        greedy_token_ids = []
        for _ in range(max_new_tokens):
            predictions_logits = []
            with torch.no_grad():
                for i in range(0, len(contexts_str), batch_size):
                    context_batch_enc = {k: v[i:i + batch_size] for k, v in contexts_enc.items()}
                    prediction_batch_logits = self.model(**context_batch_enc).logits[:, -1, :]
                    predictions_logits.append(prediction_batch_logits)

                predictions_logits = torch.cat(predictions_logits, dim=0)
                predictions_avg_logits = torch.sum(predictions_logits * weights.unsqueeze(-1), dim=0)
                greedy_token_id = torch.argmax(predictions_avg_logits)

                if greedy_token_id.item() == self.tokenizer.eos_token_id:
                    break
                greedy_token_ids.append(greedy_token_id.item())

                contexts_input_ids = torch.cat([contexts_enc['input_ids'], greedy_token_id.unsqueeze(0).repeat(len(contexts_str), 1)], dim=1).to(self.device)
                contexts_attention_mask = torch.cat([contexts_enc['attention_mask'], torch.ones((len(contexts_str), 1), dtype=torch.long, device=self.device)], dim=1)
                contexts_enc = {'input_ids': contexts_input_ids, 'attention_mask': contexts_attention_mask}

        # TODO one could check if len greedy_token_ids (when removes eos token id) == max_new_tokens, and give that as a flag of ending as well
        return greedy_token_ids


    def calculate_perplexity(self, context_batch_str, target_batch_str):
        """
        Calculates the perplexity for a batch of context and target texts.

        Args:
            input_texts (list[str]): List of input texts.
            target_texts (list[str]): List of target texts corresponding to the inputs.
        Returns:
        List[float]: Perplexity values for each input-target pair.
        """
        combined_batch_str = [context_str + target_str for context_str, target_str in zip(context_batch_str, target_batch_str)]

        combined_batch_enc = self.tokenizer(combined_batch_str, return_tensors='pt', padding=True).to(self.device)

        with torch.no_grad():
            combined_batch_logits = self.model(**combined_batch_enc).logits

        return self._calculate_perplexity_core(combined_batch_logits, combined_batch_enc, target_batch_str)

    def _calculate_perplexity_core(self, combined_batch_logits, combined_batch_enc, target_batch_str):
        """
        Computes the perplexity for each sequence in the batch.

        Args:
            logits (torch.Tensor): Logits output from the model.
            combined_batch_enc (Dict): Encoded combined inputs for reference.
            target_batch_str (list[str]): List of target texts.

        Returns:
            List[float]: Perplexity values for each target given its context.
        """
        batch_perplexity = []
        for idx, target_str in enumerate(target_batch_str):
            target_token_ids = self.tokenizer.encode(target_str, add_special_tokens=False)
            context_len = len(combined_batch_enc['input_ids'][idx]) - len(target_token_ids)
            avg_neg_log_likelihood = self._calculate_avg_neg_log_likelihood(combined_batch_logits[idx], target_token_ids, context_len)
            perplexity = torch.exp(avg_neg_log_likelihood)
            perplexity = perplexity.item() if not torch.any(torch.isinf(perplexity)) else 10e3
            batch_perplexity.append(perplexity)
        return batch_perplexity

    def _calculate_avg_neg_log_likelihood(self, logits, target_token_ids, context_len):
        """
        Calculates the average negative log likelihood for a target sequence.

        Args:
            logits (torch.Tensor): Logits for a single sequence.
            target_token_ids (list[int]): Token IDs for the target text.
            input_length (int): Length of the input portion in tokens.

        Returns:
            torch.Tensor: Average negative log likelihood for the target sequence.
        """
        log_probs = []
        for j, token_id in enumerate(target_token_ids):
            token_index = context_len + j - 1
            token_logits = logits[token_index]
            token_log_prob = F.log_softmax(token_logits, dim=-1)[token_id]
            log_probs.append(token_log_prob)
        return -sum(log_probs) / len(log_probs) if log_probs else torch.tensor(0.0)