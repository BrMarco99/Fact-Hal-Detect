import torch
import torch.nn as nn
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, StoppingCriteria, StoppingCriteriaList
from transformers import AutoConfig, BitsAndBytesConfig

import os


class LLMCompletion(nn.Module):
    def __init__(self, model_name, max_new_tokens=512, system_prompt = None):
        super(LLMCompletion, self).__init__()

        self.model_name = model_name
        self.models = {
            'llama13b': "meta-llama/Llama-2-13b-chat-hf",
            'llama7b': "meta-llama/Llama-2-7b-chat-hf",
            'vicuna13b': "lmsys/vicuna-13b-v1.5",
            'vicuna13b-16k': "lmsys/vicuna-13b-v1.5-16k",
            'openchat_3.5': "openchat/openchat_3.5",
            'Starling-LM-7B-alpha': "berkeley-nest/Starling-LM-7B-alpha",
            'Mistral': "mistralai/Mistral-7B-Instruct-v0.3",
            "llama3": "meta-llama/Meta-Llama-3-8B"
        }

        if model_name in self.models:
            model_path = self.models[model_name]
            self.generation_config = GenerationConfig.from_pretrained(model_path)
            self.generation_config.max_new_tokens = max_new_tokens
            self.generation_config.temperature = 0.
            self.generation_config.do_sample = False
            self.generation_config.top_p = 1.0
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if model_name == 'Mistral':
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, bnb_4bit_compute_dtype=torch.bfloat16, load_in_4bit=True, device_map="auto"
                ).eval()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, bnb_4bit_compute_dtype=torch.bfloat16, load_in_4bit=True, device_map="auto"
                ).eval()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    @torch.no_grad()
    def forward(self, prompt, stop_tokens=['\n'], return_prob=False):

        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        stopping_criteria = self.get_stopping_criteria(stop_tokens)
        output = self.model.generate(
            **inputs,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria = stopping_criteria,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
        response = self.tokenizer.decode(output['sequences'][0], skip_special_tokens=True)[len(prompt):]
        if return_prob:
            transition_scores = self.model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
            transition_scores = np.exp(transition_scores[0].cpu().numpy())
            input_length = inputs.input_ids.shape[1]
            generated_tokens = [self.tokenizer.decode(tok) for tok in output.sequences[0, input_length:]]
            return response, generated_tokens, transition_scores.tolist()
        else:
            return response
            
    def get_stopping_criteria(self, stop_tokens):
        truncate_length = len(self.tokenizer(f'\n')['input_ids'])
        if stop_tokens:
            stop_token_ids = [torch.LongTensor(self.tokenizer(f'\n{stop_token}')['input_ids'][truncate_length:]).cuda() for stop_token in stop_tokens]
            # define custom stopping criteria object
            class StopOnTokens(StoppingCriteria):
                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                    for stop_ids in stop_token_ids:
                        if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                            return True
                    return False
            return StoppingCriteriaList([StopOnTokens()])
        else:
            return None