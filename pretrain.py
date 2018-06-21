from vocab import Vocabulary
import cPickle as pickle 

vocabpath = 'vocab/10crop_precomp_vocab.pkl'

v = pickle.load(open(vocabpath, 'rb'))
vocab_size = len(v)

print(vocab_size)