import numpy as np
from nltk.corpus import stopwords
from nltk import download
from gensim.models import KeyedVectors
import os
download('stopwords')

class WMD:
    def __init__(self):
        self.stop_words = stopwords.words('english')
        if not os.path.exists('./data/GoogleNews-vectors-negative300.bin.gz'):
            raise ValueError("SKIP: You need to download the google news model")
        self.model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
        return

    def calc_score(self, candidate, refs):
        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        dis = []

        # split into tokens
        token_c = candidate[0].split(" ")
        token_c = [w for w in token_c if w not in self.stop_words]

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            token_r = [w for w in token_r if w not in self.stop_words]

            dis.append(self.model.wmdistance(token_c, token_r))

        mis_dis = min(dis)
        return mis_dis

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        score = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            score.append(self.calc_score(hypo, ref))

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "WMD"
