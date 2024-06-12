import pandas as pd
from collections import Counter
from nltk.util import ngrams
import math

# Load the model output results
file_path = '/Users/dezhiyu/Downloads/generated_responses.csv'  # Ensure the file path is correct
df = pd.read_csv(file_path)

# Replace NaN values with empty strings
df['answer_0'] = df['answer_0'].fillna('')
df['answer_1'] = df['answer_1'].fillna('')

# Extract reference answers and candidate answers
references = [[str(row['answer_1']).split()] for _, row in df.iterrows()]
candidates = [str(row['answer_0']).split() for _, row in df.iterrows()]

def modified_precision(references, candidate, n):
    """
    Calculate the modified n-gram precision.

    Args:
    references (list): List of reference translations.
    candidate (list): Candidate translation.
    n (int): N-gram length.

    Returns:
    tuple: Clipped count sum and total count sum.
    """
    counts = Counter(ngrams(candidate, n))
    if not counts:
        return 0, 1  # Return 0, 1 if no n-grams found
    
    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            if ngram in max_counts:
                max_counts[ngram] = max(max_counts[ngram], reference_counts[ngram])
            else:
                max_counts[ngram] = reference_counts[ngram]
    
    clipped_counts = {ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()}
    return sum(clipped_counts.values()), sum(counts.values())

def brevity_penalty(reference, candidate):
    """
    Calculate the brevity penalty.

    Args:
    reference (list): Reference translation.
    candidate (list): Candidate translation.

    Returns:
    float: Brevity penalty.
    """
    ref_length = len(reference)
    cand_length = len(candidate)
    if cand_length > ref_length:
        return 1
    elif cand_length == 0:
        return 0
    else:
        return math.exp(1 - ref_length / cand_length)

def safe_log(num, denom):
    """
    Safely compute the logarithm.

    Args:
    num (float): Numerator.
    denom (float): Denominator.

    Returns:
    float: Logarithm of num/denom, or 0 if denom is 0.
    """
    if denom == 0 or num == 0:
        return 0
    return math.log(num / denom)

def bleu_score(references, candidates, weights=[0.25, 0.25, 0.25, 0.25]):
    """
    Calculate the BLEU score for a set of candidate translations against reference translations.

    Args:
    references (list): List of reference translations.
    candidates (list): List of candidate translations.
    weights (list): Weights for n-gram precisions.

    Returns:
    float: BLEU score.
    """
    p_ns = []
    for i, candidate in enumerate(candidates):
        ref = references[i]  # Use the reference directly
        p_n = [modified_precision(ref, candidate, n) for n in range(1, len(weights) + 1)]
        p_ns.append(p_n)
    
    # Compute the geometric mean of modified precisions and apply weights
    s = (w * safe_log(p[0], p[1]) for p_n, w in zip(zip(*p_ns), weights) for p in p_n)
    bp = brevity_penalty(ref[0], candidate)
    return bp * math.exp(sum(s))

# Compute the BLEU score
bleu = bleu_score(references, candidates)
print("BLEU Score: ", bleu)
