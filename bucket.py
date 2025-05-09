import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, asdict
from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample 
import os
import json
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from utils import PROJECT_ROOT_DIR, load_json


@dataclass
class ExamplePrediction:
    """
    A single prediction for a single example -- either original or paraphrased.
    """
    example_id: str
    confidence: List[float]
    prediction: int
    gold_label: int
    example: Optional[Union[
        DefeasibleNLIExample, 
        AbductiveNLIExample, 
        ParaphrasedAbductiveNLIExample, 
        ParaphrasedDefeasibleNLIExample
    ]]
    
    @property
    def confidence_in_gold_label(self) -> float:
        return self.confidence[self.gold_label]

    @property
    def confidence_in_prediction(self) -> float:
        return np.max(self.confidence)
    
    @property
    def confidence_deviation(self) -> float:
        return abs(self.confidence_in_gold_label - 0.5)
    
    @property
    def correct(self) -> int:
        assert self.gold_label in (0, 1) and self.prediction in (0, 1)
        return int(self.prediction == self.gold_label)

@dataclass
class Bucket:
    """
    A 'bucket' is a collection of paraphrases for a single example.
    """
    original_example_id: str
    original_example_prediction: ExamplePrediction
    paraphrase_predictions: List[ExamplePrediction]
    model_name: Optional[str] = None  # model that produced the predictions
    
    @property
    def bucket_confidence_variance(self) -> float:
        return np.var([p.confidence_in_gold_label for p in self.paraphrase_predictions])
    
    @property
    def bucket_confidence_mean(self) -> float:
        return np.mean([p.confidence_in_gold_label for p in self.paraphrase_predictions])

    @property
    def bucket_confidence_std(self) -> float:
        return np.std([p.confidence_in_gold_label for p in self.paraphrase_predictions])
    
    @property
    def bucket_discrete_agreement(self) -> float:
        return len([p for p in self.paraphrase_predictions if p.prediction  == self.original_example_prediction.prediction])/len(self.paraphrase_predictions)
    
    @property
    def bucket_correctness_variance(self) -> float:
        return np.var([p.correct for p in self.paraphrase_predictions])

    @property
    def bucket_correctness_mean(self) -> float:
        return np.mean([p.correct for p in self.paraphrase_predictions])

    @property
    def bucket_confidence_shift(self) -> float:
        return self.bucket_confidence_mean - self.original_example_prediction.confidence_in_gold_label
    
    @property
    def bucket_paraphrase_accuracy(self) -> float:
        predictions = []
        ground_truth = []
        for paraphrase in self.paraphrase_predictions:
            predictions.append(paraphrase.prediction)
            ground_truth.append(paraphrase.gold_label)
       
        return accuracy_score(ground_truth, predictions)
