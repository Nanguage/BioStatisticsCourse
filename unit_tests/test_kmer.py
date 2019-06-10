from seqlearn.kmer import *


def test_vocab():
    voc = Vocabulary(k=6)
    assert voc.table.shape[0] == 4096
    assert voc.kmer_to_idx("AAAAAA") == 0
    assert voc.kmer_to_idx("AAAAAT") == 3
    assert voc.kmer_to_idx("TTTTTT") == 4095
    assert voc.idx_to_kmer(100) == 'AACGCA'
    assert voc.idx_to_kmer(0) == 'AAAAAA'
    assert voc.idx_to_kmer(4095) == "TTTTTT"


def test_iter_kmers():
    assert list( iter_kmers("ACCTA", 2, 1) ) == ['AC', 'CC', 'CT', 'TA']
    assert list( iter_kmers("ACCTA", 2, 2) ) == ['AC', 'CT']
    assert list( iter_kmers("ACCTA", 3, 1) ) == ['ACC', 'CCT', 'CTA']


def test_kmer_count():
    voc = Vocabulary(k=3)
    assert voc.kmer_count("AAAAA", 1)[0] == 3
    kmer_cnt = KmerCount(voc, 1)
    assert kmer_cnt.count("AAAAA")[0] == 3
    assert kmer_cnt.count("TTTTT")[-1] == 3


def test_kmer_sent():
    voc = Vocabulary(k=3)
    assert voc.get_kmer_sentence("AAAAA", 1) == [0, 0, 0]
    assert voc.get_kmer_sentence("TTTTT", 1) == [63, 63, 63]


def test_expand_unknow():
    voc = Vocabulary(k=3)
    assert voc.expand_unknow("N") == ["A", "C", "G", "T"]
    assert voc.expand_unknow("ANA") == ["AAA", "ACA", "AGA", "ATA"]
    assert voc.expand_unknow("NN") == [i+j for i in "ACGT" for j in "ACGT"]
