from Levenshtein import setratio, seqratio
from nltk.tokenize import word_tokenize
from datasets import load_metric
from abductive_data import ParaphrasedAbductiveNLIExample, AbductiveNLIExample
from defeasible_data import ParaphrasedDefeasibleNLIExample, DefeasibleNLIExample
import string
import datasets
import os
from utils import PROJECT_ROOT_DIR

syntdiv = load_metric(os.path.join(PROJECT_ROOT_DIR, 'syntdiv_metric.py'), experiment_id=0)
semantic_similarity = load_metric(os.path.join(PROJECT_ROOT_DIR, 'bleurt_metric.py'), experiment_id=0)

datasets.logging.set_verbosity_error()
datasets.logging.disable_progress_bar() 

def clean_paraphrase(p: str) -> str:
    """
    Lower case, remove surrounding whitespace, remove punctuation 
    """
    return p.strip().lower().translate(str.maketrans('', '', string.punctuation))

def _lexical_diversity_score(prediction, reference):
    pred, ref = word_tokenize(prediction.lower()), word_tokenize(reference.lower())
    set_diversity = 1 - setratio(pred, ref)
    return set_diversity
    
    #seq_diversity = 1 - seqratio(pred, ref)
    # return #(set_diversity + seq_diversity) / 2 

def _syntactic_diversity_score(predictions, references):
    return syntdiv.compute(predictions=predictions, references=references)["scores"]

def _bleurt_score(predictions, references):
    return semantic_similarity.compute(predictions=predictions, references=references)["scores"]

def get_syntactic_diversity_score(paraphrase):
    assert type(paraphrase) in [ParaphrasedAbductiveNLIExample, ParaphrasedDefeasibleNLIExample]
    if isinstance(paraphrase, ParaphrasedAbductiveNLIExample):
        original_example = AbductiveNLIExample(**paraphrase.original_example) if isinstance(paraphrase.original_example, dict) else paraphrase.original_example
        references = [clean_paraphrase(original_example.hyp1), clean_paraphrase(original_example.hyp2)]
        scores = _syntactic_diversity_score(
            predictions=[clean_paraphrase(paraphrase.hyp1_paraphrase), clean_paraphrase(paraphrase.hyp2_paraphrase)],
            references=references
        )
        return (scores[0] + scores[1]) / 2
    elif isinstance(paraphrase, ParaphrasedDefeasibleNLIExample):
        original_example = DefeasibleNLIExample(**paraphrase.original_example) if isinstance(paraphrase.original_example, dict) else paraphrase.original_example
        return _syntactic_diversity_score(predictions=[clean_paraphrase(paraphrase.update_paraphrase)], references=[clean_paraphrase(original_example.update)])[0]
    

def get_lexical_diversity_score(paraphrase):
    assert type(paraphrase) in [ParaphrasedAbductiveNLIExample, ParaphrasedDefeasibleNLIExample]
    
    if isinstance(paraphrase, ParaphrasedAbductiveNLIExample):
        original_example = AbductiveNLIExample(**paraphrase.original_example) if isinstance(paraphrase.original_example, dict) else paraphrase.original_example
        h1 = _lexical_diversity_score(clean_paraphrase(paraphrase.hyp1_paraphrase), clean_paraphrase(original_example.hyp1))
        h2 = _lexical_diversity_score(clean_paraphrase(paraphrase.hyp2_paraphrase), clean_paraphrase(original_example.hyp2))
        return (h1 + h2) / 2
    elif isinstance(paraphrase, ParaphrasedDefeasibleNLIExample):
        original_example = DefeasibleNLIExample(**paraphrase.original_example) if isinstance(paraphrase.original_example, dict) else paraphrase.original_example
        return _lexical_diversity_score(
            clean_paraphrase(paraphrase.update_paraphrase), 
            clean_paraphrase(original_example.update)
        )

def get_semantic_similarity_score(paraphrase):
    assert type(paraphrase) in [ParaphrasedAbductiveNLIExample, ParaphrasedDefeasibleNLIExample]
    
    if isinstance(paraphrase, ParaphrasedAbductiveNLIExample):
        original_example = AbductiveNLIExample(**paraphrase.original_example) if isinstance(paraphrase.original_example, dict) else paraphrase.original_example
        h1 = _bleurt_score([clean_paraphrase(paraphrase.hyp1_paraphrase)], [clean_paraphrase(original_example.hyp1)])[0]
        h2 = _bleurt_score([clean_paraphrase(paraphrase.hyp2_paraphrase)], [clean_paraphrase(original_example.hyp2)])[0]
        return (h1 + h2) / 2
    elif isinstance(paraphrase, ParaphrasedDefeasibleNLIExample):
        original_example = DefeasibleNLIExample(**paraphrase.original_example) if isinstance(paraphrase.original_example, dict) else paraphrase.original_example
        return _bleurt_score(
            [clean_paraphrase(paraphrase.update_paraphrase)], 
            [clean_paraphrase(original_example.update)]
        )[0]