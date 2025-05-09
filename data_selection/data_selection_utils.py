import numpy as np
from typing import List, Tuple, Any, Dict
import random
from collections import defaultdict


def float_floor(num, precision=1):
    if num == 1.0:
        return 0.9
    return np.true_divide(np.floor(num * 10**precision), 10**precision)

def stratify_examples_by_range(examples: List[Tuple[Any, float]], print_ranges=True) -> Dict[float, List[Any]]:
    """
    Given a list of (example, score) inputs, stratifies according to their score
    """
    random.seed(42)
    ranges = defaultdict(list) # {0.1: [e1, e2], 0.2: [e3, e4]}
    for example_id, score in examples:
        ranges[float_floor(score, precision=1)].append((example_id, score))
    
    if print_ranges:
        for k, v in ranges.items():
            print(k, len(v))
    return ranges
