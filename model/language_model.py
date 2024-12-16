import torch

class LanguageModel:
    """
    Encapsulates a transformer-based language model for text generation. This class provides functionalities
    to generate text based on input prompts.

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