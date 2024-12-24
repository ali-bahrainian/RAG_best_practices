import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

import sys
sys.path.append("mixtral-offloading")
from transformers import AutoConfig
from src.build_model import OffloadConfig, QuantConfig, build_model
from hqq.core.quantize import BaseQuantizeConfig

class ModelLoader:
    """
    Responsible for loading a specific language model and its associated tokenizer.

    Attributes:
        model_name (str): The name of the loaded model.
        model_type (str): The type of the model ('causal', 'seq2seq', 'classification').
        model (transformers.PreTrainedModel): The loaded language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
    """

    def __init__(self, model_name, model_type, quant_type=None):
        """
        Initializes the ModelLoader with a specified model name, model type, and optional quantization.

        Args:
            model_name (str): The name of the model to be loaded.
            model_type (str): The type of the model to be loaded ('causal', 'seq2seq').
            quant_type (str, optional): Type of quantization ('8bit', '4bit', or None).
        """
        self.model_name = model_name
        self.model_type = model_type
        if self.model_name != 'mistralai/Mixtral-8x7B-Instruct-v0.1':
            # Generate quantization configuration
            bnb_config = None
            if quant_type == '8bit':
                bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
            elif quant_type == '4bit':
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        
            # Load the model based on the type
            model_loader_function = {
                'causal': AutoModelForCausalLM,
                'seq2seq': AutoModelForSeq2SeqLM
            }.get(model_type)
        
            if not model_loader_function:
                raise ValueError(f"Unsupported model type: {model_type}")
        
            # Prepare kwargs for model loading
            model_kwargs = {
                'pretrained_model_name_or_path': self.model_name,
                'quantization_config': bnb_config,
            }
        
            if model_type == "seq2seq":
                model_kwargs['device_map'] = 'auto'
        
            self.model = model_loader_function.from_pretrained(**model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
            
        else:
            print('Instruct45B \n---')
            model_name = self.model_name
            model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",  
                        offload_folder="offload",  
                    )
            self.model = model
            self.tokenizer= AutoTokenizer.from_pretrained(self.model_name, padding_side='left')