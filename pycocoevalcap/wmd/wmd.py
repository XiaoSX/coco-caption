import numpy as np
from nltk.corpus import stopwords
from nltk import download
from gensim.models import KeyedVectors
from pyemd import emd
from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

download('stopwords')

# get n-gram token for one sentence
def precook(s, n=1, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    counts = defaultdict(int)
    for k in xrange(1,n+1):
        for i in xrange(len(s)-k+1):
            ngram = tuple(s[i:i+k])[0]
            counts[ngram] += 1
    return counts

class WMD:
    def __init__(self):
        self.stop_words = stopwords.words('english')
        if not os.path.exists('/home/renmeng/gitlab/wmd/data/GoogleNews-vectors-negative300.bin.gz'):
            raise ValueError("SKIP: You need to download the google news model")
        self.model = KeyedVectors.load_word2vec_format('/home/renmeng/gitlab/wmd/data/GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.test = []
        self.refs = []
        self.ctest = []
        self.crefs = []
        self.document_frequency = defaultdict(float)
        return

    # tokenize for one sentence, remove stop words
    def preprocess(self, candidate, refs):
        assert (len(candidate) > 0)
        assert (len(refs) > 0)

        # split into tokens
        token_c = candidate.split(" ")
        token_c = [w for w in token_c if w not in self.stop_words]

        token_r = []
        for reference in refs:
            # split into tokens
            token_r_tmp = reference.split(" ")
            token_r_tmp = [w for w in token_r_tmp if w not in self.stop_words]

            token_r.append(token_r_tmp)

        return token_c, token_r

    def compute_doc_freq(self):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.iteritems()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def bulid_wmd(self, hypo, ref):
        hypo, ref = self.preprocess(hypo, ref)
        self.test.append(hypo)
        self.refs.append(ref)
        self.crefs.append([precook(ref_, 1) for ref_ in ref])
        self.ctest.append(precook(hypo, 1, True))

    def wmdistance(self, document1, document2, tf_idf1, tf_idf2, weight):

        document1 = [token for token in document1 if self.model.__contains__(token)]
        document2 = [token for token in document2 if self.model.__contains__(token)]
        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)

        # Compute distance matrix.

        distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if t1 not in docset1 or t2 not in docset2:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = np.sqrt(np.sum((self.model.get_vector(t1) - self.model.get_vector(t2)) ** 2))

        if np.sum(distance_matrix) == 0.0:
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def _tf_idf(word_tfidf):
            d = np.zeros(vocab_len, dtype=np.double)
            for id_, term in dictionary.items():
                d[id_] = word_tfidf[term]
            return d

        def nbow(document):
            d = np.zeros(vocab_len, dtype=np.double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        if weight == 'TFIDF':
            d1 = _tf_idf(tf_idf1)
            d2 = _tf_idf(tf_idf2)
        elif weight == 'Norm':
            d1 = nbow(document1)
            d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)

    def score_(self, weight):
        def counts2vec(cnts):
            vec = defaultdict(float)
            for (ngram, term_freq) in cnts.iteritems():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[ngram] = float(term_freq) * (ref_len - df)

            return vec

        ref_len = np.log(float(len(self.crefs)))
        scores = []
        for i in range(len(self.crefs)):
            test_, refs_ = self.test[i], self.refs[i]
            ctest_, crefs_ = self.ctest[i], self.crefs[i]
            vec = counts2vec(ctest_)
            score_ = []
            for j in range(len(crefs_)):
                ref, ref_text = crefs_[j], refs_[j]
                vec_f = counts2vec(ref)
                score_.append(self.wmdistance(test_, ref_text, vec, vec_f, weight))
            scores.append(np.min(score_))
        return np.mean(scores), scores

    def compute_score(self, gts, res, weight='TFIDF'):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            self.bulid_wmd(hypo[0], ref)
        self.compute_doc_freq()

        (score, scores) = self.score_(weight)

        return score, scores

    def method(self):
        return "WMD"
