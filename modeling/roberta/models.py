from abc import ABC, abstractmethod
import numpy
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from typing import List, Union, Tuple
from itertools import chain
import numpy as np


class TrainedModel:
    
    def __init__(self, trained_model_dir: str, cache_dir: str = 'checkpoints/hf_cache', multiple_choice=True) -> None:
        self.is_multiple_choice = multiple_choice
        self.tokenizer = TrainedModel.get_tokenizer(trained_model_dir, cache_dir)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.trained_model = self.get_model(trained_model_dir, cache_dir, multiple_choice=self.is_multiple_choice)
        
    @abstractmethod
    def predict(self) -> numpy.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_tokenizer(model_path: str, hf_cache_path: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=hf_cache_path,
            use_fast=False,
            revision="main",
            use_auth_token=False,
        )

    def get_model(self, model_path: str, hf_cache_path: str, multiple_choice=True) -> Union[AutoModelForMultipleChoice, AutoModelForSequenceClassification]:
        print('Loading model from %s' % model_path)
        model_wrapper = AutoModelForMultipleChoice if multiple_choice else AutoModelForSequenceClassification
        return model_wrapper.from_pretrained(
            model_path,
            from_tf=False,
            cache_dir=hf_cache_path
        ).to(device=self.device)


class AbductiveTrainedModel(TrainedModel):
    NUM_CHOICES = 2

    def get_example_embedding(self, obs1: str, obs2: str, hyp1: str, hyp2: str) -> numpy.ndarray:
        first_sentences = [[obs1] * AbductiveTrainedModel.NUM_CHOICES for context in [obs1]]
        second_sentences = [
            [f"{h} {obs2}" for h in [hyp1, hyp2]]
        ]
        
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))
        
        result = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=1024,
            padding=True,
        )
        result = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in result.items()}
        
        outputs = self.trained_model.roberta(
            input_ids=torch.tensor(result['input_ids']).to(device=self.device), 
            attention_mask=torch.tensor(result['attention_mask']).to(device=self.device)
        )
        sequence_output = outputs[0]
        cls_rep = sequence_output[:, 0, :]
        return cls_rep.squeeze(0).detach().cpu().numpy() # take <s> token (equiv. to [CLS])


    def predict(self, obs1: str, obs2: str, hyp1: str, hyp2: str) -> numpy.ndarray:
        return self._get_prediction(obs1, obs2, hyp1, hyp2)

    def _get_prediction(self, obs1, obs2, hyp1, hyp2) -> numpy.ndarray:
        """
        Input will either be a single string in case of partial input, or 
        tuple of strings in the case of full input.
        """
        first_sentences = [[obs1] * AbductiveTrainedModel.NUM_CHOICES for context in [obs1]]
        second_sentences = [
            [f"{h} {obs2}" for h in [hyp1, hyp2]]
        ]
        
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))
        
        result = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=1024,
            padding=True,
        )
        result = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in result.items()}
        
        outputs = self.trained_model(
            input_ids=torch.tensor(result['input_ids']).to(device=self.device), 
            attention_mask=torch.tensor(result['attention_mask']).to(device=self.device)
        )
        probs = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
        return probs[0]


class DefeasibleTrainedModel(TrainedModel):
    
    def get_example_embedding(self, premise: str, hypothesis: str, update: str) -> numpy.ndarray:
        result = self.tokenizer(
            [(premise + hypothesis, update)],
            padding="max_length", 
            max_length=128, 
            truncation=True, 
            return_tensors="pt"
        ).to(device=self.device)
        outputs = self.trained_model.roberta(**result)
        sequence_output = outputs[0]
        cls_rep = sequence_output[:, 0, :]
        return cls_rep.squeeze(0).detach().cpu().numpy() # take <s> token (equiv. to [CLS])

    def predict(self, premise: str, hypothesis: str, update: str) -> numpy.ndarray:
        return self._get_prediction((premise + hypothesis, update))

    def _get_prediction(self, inp: Union[str, Tuple[str, str]]) -> numpy.ndarray:
        result = self.tokenizer(
            [inp],
            padding="max_length", 
            max_length=128, 
            truncation=True, 
            return_tensors="pt"
        ).to(device=self.device)
        outputs = self.trained_model(**result)
        probs = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
        return probs[0]
    
class DefeasibleTrainedPartialInputModel(TrainedModel):
    
    def predict(self, update: str) -> numpy.ndarray:
        return self._get_prediction(update)

    def _get_prediction(self, inp: Union[str, Tuple[str, str]]) -> numpy.ndarray:
        result = self.tokenizer(
            [inp],
            padding="max_length", 
            max_length=128, 
            truncation=True, 
            return_tensors="pt"
        ).to(device=self.device)
        outputs = self.trained_model(**result)
        probs = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
        return probs[0]


class PretrainedNLIModel:

    def __init__(self, trained_model_dir: str, cache_dir: str = 'checkpoints/hf_cache', label_map = ['E', 'N', 'C']) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = PretrainedNLIModel.get_tokenizer(trained_model_dir, cache_dir)
        self.trained_model = self.get_model(trained_model_dir, cache_dir)
        self.label_map = label_map
        
    def predict(self, premise: str, hypothesis: str) -> np.ndarray:
        return self._get_prediction((premise, hypothesis))

    def predict_label(self, premise: str, hypothesis: str) -> str:
        prediction = self.predict(premise, hypothesis)
        return self.label_map[np.argmax(prediction)]

    def _get_prediction(self, inp: Union[str, Tuple[str, str]]) -> np.ndarray:
        """
        Input will either be a single string in case of partial input, or 
        tuple of strings in the case of full input.
        """
        result = self.tokenizer(
            [inp],
            padding="max_length", 
            max_length=128, 
            truncation=True, 
            return_tensors="pt"
        )
        outputs = self.trained_model(
            input_ids=torch.tensor(result['input_ids']).to(device=self.device), 
            attention_mask=torch.tensor(result['attention_mask']).to(device=self.device)
        )
        probs = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
        return probs[0]

    @staticmethod
    def get_tokenizer(model_path: str, hf_cache_path: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=hf_cache_path,
            use_fast=False,
            revision="main",
            use_auth_token=False,
        )

    def get_model(self, model_path: str, hf_cache_path: str) -> AutoModelForSequenceClassification:
        print('Loading model from %s' % model_path)
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            from_tf=bool(".ckpt" in model_path),
            cache_dir=hf_cache_path
        ).to(device=self.device)