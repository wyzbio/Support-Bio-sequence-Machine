# -*- encoding: utf-8 -*-
# @Time     :   2023/03/15 23:21:46
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   None

import Levenshtein
from Bio.Align import PairwiseAligner
from functools import partial
from multiprocessing import Pool
from _mini_tools import *


#------------ Levenshtein距离 ------------#
def levenshetein_train_similarity(sequence_pair):
    seq1, seq2 = sequence_pair
    return Levenshtein.distance(seq1, seq2)

def levenshetein_test_similarity(query, target_list):
    similarities = []
    for target in target_list:
        similarities.append(Levenshtein.distance(query, target))
    return similarities


def compute_levenshetein_train_kernel(sequences, process_num):
    num_sequences = len(sequences)
    similarity_matrix = [[0] * num_sequences for _ in range(num_sequences)]
    sequence_pairs = [(sequences[i], sequences[j]) for i in range(num_sequences) for j in range(i, num_sequences)]
    with Pool(processes=process_num) as pool:
        scores = pool.map(levenshetein_train_similarity, sequence_pairs)
    for i in range(num_sequences):
        for j in range(i, num_sequences):
            index = i * num_sequences + j - (i * (i + 1)) // 2
            similarity_matrix[i][j] = similarity_matrix[j][i] = scores[index]
    return similarity_matrix


def compute_levenshetein_test_kernel(queries, targets, process_num):
    similarity_matrix = [[None]*len(targets) for _ in range(len(queries))]
    with Pool(processes=process_num) as pool:
        results = []
        for i, query in enumerate(queries):
            result = pool.apply_async(levenshetein_test_similarity, args=(query, targets))
            results.append((i, result))
        for i, result in results:
            similarities = result.get()
            for j, similarity in enumerate(similarities):
                similarity_matrix[i][j] = similarity
    return similarity_matrix


if __name__ == "__main__":
    
    pass