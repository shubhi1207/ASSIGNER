import os, sys, glob, json
import numpy as np
import argparse
import torch
import ast
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
from bert_score import score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

def clean_text(text):
    # Remove invalid characters, encode to utf-8, and decode back
    return text.encode('utf-8', 'ignore').decode('utf-8')

def distinct_n_gram(hypn, n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = list(nltk.ngrams(hyp.split(), n))
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(set(hyp_ngrams))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams / total_ngrams)
    return np.mean(dist_list)

def distinct_n_gram_inter_sent(hypn, n):
    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(set(hyp_ngrams))
    if total_ngrams == 0:
        return 0
    return unique_ngrams / total_ngrams

def get_bleu(recover, reference):
    recover_clean = clean_text(recover.strip())
    reference_clean = clean_text(reference.strip())
    
    # Avoid division by zero if recover is empty
    if len(recover_clean) == 0:
        return 0.0
    
    return sentence_bleu([reference_clean.split()], recover_clean.split(), 
                         smoothing_function=SmoothingFunction().method4)

references = []
recovers = []
bleu = []
rougel = []
avg_len = []
dist1 = []
sentenceDict = {}
referenceDict = {}
sourceDict = {}

for i in range(3280):
    sentenceDict[i] = []
    referenceDict[i] = []
    sourceDict[i] = []

# Update this path with your actual file path

path = 'path_to_output.txt'

with open(path, 'r') as f:
    cnt = 0
    for row in f:
        values = ast.literal_eval(row.strip())
        recover = values[0].strip()
        reference = values[1].strip()

        # Append to lists
        references.append(reference)
        recovers.append(recover)
        avg_len.append(len(recover.split(' ')))

        # Calculate BLEU score
#         bleu.append(get_bleu(recover, reference))

        # Distinct n-gram
        dist1.append(distinct_n_gram([recover], 1))

        # Store in dictionaries
        sentenceDict[cnt].append(recover)
        referenceDict[cnt].append(reference)
        cnt += 1

# Clean the texts before BERT score calculation
recovers_clean = [clean_text(r) for r in recovers]
references_clean = [clean_text(r) for r in references]

# Calculate BERT score
P, R, F1 = score(recovers_clean, references_clean, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)

# Print the results
print('*' * 30)

# print('avg ROUGE-L score:', np.mean(rougel))  # Uncomment if you calculate ROUGE
print('avg BERT score:', torch.mean(F1))
print('avg Distinct-1 score:', np.mean(dist1))
print('avg sentence length:', np.mean(avg_len))
