import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

import sys
sys.path.append("mixtral-offloading")
from transformers import AutoConfig
from src.build_model import OffloadConfig, QuantConfig, build_model
from hqq.core.quantize import BaseQuantizeConfig

HF_TOKEN = ''

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
                'token': HF_TOKEN,
                'quantization_config': bnb_config,
            }
        
            if model_type == "seq2seq":
                model_kwargs['device_map'] = 'auto'
        
            self.model = model_loader_function.from_pretrained(**model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN, padding_side='left')
            
        else:
            print('MoE\n---')
            model_name = self.model_name
            quantized_model_name = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
            state_path = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

            config = AutoConfig.from_pretrained(quantized_model_name)

            offload_per_layer = 4
            num_experts = config.num_local_experts

            offload_config = OffloadConfig(
            main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
            offload_size=config.num_hidden_layers * offload_per_layer,
            buffer_size=4,
            offload_per_layer=offload_per_layer,
            )
        
            attn_config = BaseQuantizeConfig(
                nbits=4,
                group_size=64,
                quant_zero=True,
                quant_scale=True,
            )
            attn_config["scale_quant_params"]["group_size"] = 256
            
            ffn_config = BaseQuantizeConfig(
                nbits=2,
                group_size=16,
                quant_zero=True,
                quant_scale=True,
            )
            quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)
            
            model = build_model(
                device=torch.device("cuda:0"),
                quant_config=quant_config,
                offload_config=offload_config,
                state_path=state_path
            )
            self.model = model
            self.tokenizer= AutoTokenizer.from_pretrained(self.model_name, token=HF_TOKEN, padding_side='left')