#!/usr/bin/python3

import nltk
import logging
import treetaggerwrapper
# treetaggerwrapper available on pypi (version tested: 2.2.2)
import pickle
import argparse
from ap_read import read_data
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_TEST_DATA = 'texts-en.pkl'
DEFAULT_PICKLE_JAR = 'texts-en_POS.pkl'

class POSTransformer(BaseEstimator, TransformerMixin):
	#transforms data to corresponding POS tags
    def __init__(self):
        pass
    def fit(self, raw_documents, y=None):
        return self
    def transform(self, raw_documents):
        return self.fit_transform(raw_documents)
    def fit_transform(self, raw_documents, y=None):
        return getPOSTags(raw_documents)


def pos_tag(inp:str, out, tagdir:str='/usr/local/tree-tagger'):
    #Generates POS representation of data and pickles output into a jar.
    texts, genders, ages = read_data(inp)
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR=tagdir, TAGOPT='-token -sgml')
    logging.info('POS tagging data')
    
    pos_texts = []
    d_infothresholds = {int((i/100.0*len(texts))):"%i%%"%(i) for i in range(0, 101)}
    for i, t in enumerate(texts):
        tags = [el.split('\t')[1] for el in tagger.tag_text(t) if len(el.split()) == 2]
        pos_texts.append(' '.join(tags))
        if i in d_infothresholds.keys():
            logging.info('{} of documents processed'.format(d_infothresholds[i]))
            
    logging.info('Pickling results to {}'.format(out.name))
    pickle.dump((pos_texts, genders, ages), out)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s  %(levelname)s:%(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', type=str, default=DEFAULT_TEST_DATA, help='path to pickled PAN data')
    parser.add_argument('-o', '--out', type=argparse.FileType('wb'), default=None, help='destination for pickled POS data')
    args = parser.parse_args()

    out = args.out or open(DEFAULT_PICKLE_JAR, 'wb')
    pos_tag(args.inp, out)
