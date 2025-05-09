import fasttext
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import Union, Any, List, Tuple
from tqdm import tqdm
import numpy as np

from abductive_data import AbductiveNLIDataset, anli_dataset, AbductiveNLIExample, ParaphrasedAbductiveNLIExample
from defeasible_data import DefeasibleNLIDataset, DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample


ANLI = "__label__{label} {obs1} {hyp1} {obs2} | {obs1} {hyp2} {obs2}"



class FastTextClassifier():
    
    def __init__(self, example_format_function, partial_input=False):
        self.example_format_function = example_format_function
        self.partial_input = partial_input

    def preprocess_example(self, e: Union[AbductiveNLIExample, DefeasibleNLIExample, ParaphrasedDefeasibleNLIExample, ParaphrasedAbductiveNLIExample]):
        formatted_example = self.example_format_function(e, self.partial_input).lower()
        return formatted_example

    def train(self, train_examples: List[Union[DefeasibleNLIExample, AbductiveNLIExample]], train_fname: str):
        with open(train_fname, 'w') as outfile:
            for example in tqdm(train_examples):
                e = self.preprocess_example(example)
                outfile.write("%s\n" % e)

        self.model = fasttext.train_supervised(input=train_fname, wordNgrams=4, epoch=5)

    def infer(self, example: Union[DefeasibleNLIExample, AbductiveNLIExample]):
        """
        Warning: Assumes binary classification!
        """
        prepped_example = self.preprocess_example(example)
        prediction, score = self.model.predict(prepped_example)
        assert len(prediction) == 1
        assert len(score) == 1

        prediction = int(prediction[0].split('__label__')[1])

        confidences = np.zeros(2)
        confidences[prediction] = score
        confidences[abs(prediction-1)] = 1.0 - score

        return prediction, confidences
    