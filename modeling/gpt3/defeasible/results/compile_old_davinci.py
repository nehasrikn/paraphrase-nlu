from utils import load_jsonlines, PROJECT_ROOT_DIR, write_json
import os
from modeling.gpt3.gpt3 import extract_answer, extract_confidences

snli_davinci = load_jsonlines(os.path.join(PROJECT_ROOT_DIR, 'modeling/gpt3/defeasible/results/snli_human_semantic_equivalence_davinci_002.jsonl'))


snli_human = {}

for bucket in snli_davinci:
    bucket_dict = {
        'original_confidence': extract_confidences(bucket['bucket_predictions']['original_prediction']),
        'original_prediction': extract_answer(bucket['bucket_predictions']['original_prediction']),
        'bucket_confidences': [],
        'gold_label': bucket['bucket_predictions']['label']
    }
    
    for p in bucket['bucket_predictions']['bucket_preds']:
        if not extract_confidences(p):
            continue
        bucket_dict['bucket_confidences'].append({
            'confidence': extract_confidences(p),
            'prediction': extract_answer(p),
            'paraphrased_example': None
        })
    
    
    snli_human[bucket['example_id']] = bucket_dict

write_json(snli_human, os.path.join(PROJECT_ROOT_DIR, f'modeling/gpt3/defeasible/results/snli_human_semantic_equivalence_davinci_002_buckets.json'))
