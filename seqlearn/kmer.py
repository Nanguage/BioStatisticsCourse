from collections import Counter
from typing import Iterator, List

import numpy as np


NUC_CHAR_SET = "ATCG"
UNKNOW_CHAR = 'N'


class Vocabulary(object):
    def __init__(self, k:int=6, char_set:str=NUC_CHAR_SET, unknow_char:str=UNKNOW_CHAR):
        self.k = k
        self.char_set = sorted(char_set)
        self.unknow_char = unknow_char
        self.table = self.__make_table()

    def __make_table(self) -> "numpy.ndarray":
        table_ = []
        def permutation(seq):
            if len(seq) == self.k:
                table_.append(seq)
            else:
                for c in self.char_set:
                    permutation(seq+c)
        permutation("")
        table = np.array(table_)
        table.sort()
        return table

    def kmer_to_idx(self, kmer:str) -> int:
        assert kmer in self.table
        return self.table.searchsorted(kmer)

    def idx_to_kmer(self, idx:int) -> str:
        assert idx < len(self.table)
        return self.table[idx]

    def kmer_count(self, seq:str, stride:int) -> "numpy.ndarray":
        kcnt = KmerCount(self, stride)
        return kcnt.count(seq)

    def get_kmer_sentence(self, seq:str, stride:int) -> List[int]:
        sent = []
        for kmer in iter_kmers(seq, self.k, stride):
            sent.append(self.kmer_to_idx(kmer))
        return sent

    def expand_unknow(self, seq:str) -> List[str]:
        expanded = []
        def expand(seq):
            if seq.count(self.unknow_char) == 0:
                expanded.append(seq)
            else:
                for c in self.char_set:
                    idx_un = seq.index(self.unknow_char)
                    expand(seq[:idx_un]+c+seq[idx_un+1:])
        expand(seq)
        return expanded


def iter_kmers(seq:str, k:int, stride:int) -> Iterator[str]:
    for start in range(len(seq))[::stride]:
        end = start + k
        kmer = seq[start:end]
        if len(kmer) == k:
            yield kmer


class KmerCount(object):
    def __init__(self, vocab:"Vocabulary", stride:int=1):
        self.vocab = vocab
        self.k = vocab.k
        self.stride = stride
        self.table = np.zeros(vocab.table.shape[0])

    def count(self, seq:str) -> "numpy.ndarray":
        seq = seq.upper()
        for kmer in iter_kmers(seq, self.k, self.stride):
            if self.vocab.unknow_char not in kmer:
                idx = self.vocab.kmer_to_idx(kmer)
                self.table[idx] += 1
            else:
                for kmer_exp in self.vocab.expand_unknow(kmer):
                    idx = self.vocab.kmer_to_idx(kmer_exp)
                    self.table[idx] += 1
        return self.table

