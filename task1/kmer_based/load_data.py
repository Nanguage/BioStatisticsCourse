import numpy as np
from Bio.SeqIO import FastaIO
from tqdm import tqdm

from seqlearn.kmer import Vocabulary


def load_data(k, stride, pos_fasta, neg_fasta):
    vocab = Vocabulary(k=k)

    X = []
    n_pos = 0; n_neg = 0
    for fasta in pos_fasta, neg_fasta:
        with open(fasta) as f:
            for s in tqdm(FastaIO.FastaIterator(f)):
                seq = str(s.seq)
                if vocab.unknow_char in seq:
                    continue
                try:
                    x = vocab.kmer_count(seq, stride)
                except AssertionError:
                    continue
                X.append(x)
                if fasta == pos_fasta:
                    n_pos += 1
                else:
                    n_neg += 1

    X = np.vstack(X)
    y = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
    return X, y