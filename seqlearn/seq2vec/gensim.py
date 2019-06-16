from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import Iterable
from tqdm import tqdm

from seqlearn.kmer import Vocabulary


class Seq2Vec(object):

    def __init__(self, model:"gensim.Doc2Vec"=None,
                 vec_size=100, k=6, stride=1):
        self.vocab = Vocabulary(k)
        self.k = k
        self.stride = stride
        self.vec_size = vec_size
        self.model = model

    def build_from_seqs(self, seq_iter:Iterable[str]):
        documents =  []
        for i, seq in tqdm(enumerate(seq_iter)):
            try:
                s = self.vocab.get_kmer_sentence(seq, self.stride)
                s = [str(i) for i in s]
            except AssertionError:
                continue
            doc = TaggedDocument(s, [i])
            documents.append(doc)
        self.documents = documents

    def train_model(self, kw_doc2vec={}):
        kw_ = {'window':10, 'min_count':1, 'workers':8}
        kw_.update(kw_doc2vec)
        self.model = Doc2Vec(self.documents, vector_size=self.vec_size, **kw_)

    def save(self, fname):
        self.model.save(fname)

    def load(self, fname):
        self.model = Doc2Vec.load(fname)

    def seq_to_vec(self, seq:str):
        s = self.vocab.get_kmer_sentence(seq, self.stride)
        s = [str(i) for i in s]
        vec = self.model.infer_vector(s)
        return vec

