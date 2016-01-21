#!/usr/bin/python3

"""
Data management module for extracting and serializing custom features from input data
"""

# TODO: maybe implement function to postprocess extracted features (tfidf transformation for ngrams, feature selection for ngrams etc.)

import logging
import argparse
from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

from ap_read import read_data
from getPOSTags import POSTransformer
from style_features import StyleVectorizer
from readability_features import FleschReadVectorizer

DEFAULT_TEST_DATA = 'texts-en_head.xml'
DEFAULT_TOKEN_JAR = 'texts-en.pkl'
DEFAULT_POS_JAR = 'texts-en_POS.pkl'

FEAT_DICT = {'ngram':CountVectorizer,  'style':StyleVectorizer, 'read':FleschReadVectorizer}


def extract_feats(data:str, feat_name:str, outpath:str, pos:str=False, **kwargs):
    """Extracts features from input data and writes them to disk
    """
    # input correction
    if outpath[-1] != '/': path = outpath + '/'
    else: path = outpath
    if pos and data == DEFAULT_TOKEN_JAR:
        src = DEFAULT_POS_JAR
    else: src = data
        
    texts, genders, ages = read_data(src)
    vect = FEAT_DICT[feat_name](**kwargs)

    if feat_name == 'ngram':
        logging.info('Generating ngram feature vectors (range:{}, max_features:{}, min_df:{}, max_df:{})'.format(kwargs['ngram_range'], str(kwargs['max_features']), str(kwargs['min_df']), str(kwargs['max_df'])))
    else:
        logging.info('Generating {} feature vectors'.format(feat_name))
    feats = vect.fit_transform(texts)

    logging.info('Writing feature matrix to disk')

    if feat_name == 'ngram':
        if pos:
            feat = 'pos_' + feat_name
        else:
            feat = feat_name
        fn = 'feat_matr_{}_n{}_{}'.format(feat,
                                         '-'.join(str(n) for n in kwargs['ngram_range']),
                                         datetime.now().strftime('%m%d_%H%M%S'))
    else:
        fn = 'feat_matr_{}_{}'.format(feat_name, datetime.now().strftime('%m%d_%H%M%S'))
    np.save(path + fn, feats)

    # write parameters (if any) to log file to avoid ridiculous file names
    if kwargs:
        with open(path + 'feats.log', 'a') as op:
            op.write('\n{} {} {}\t{}'.format(datetime.now().strftime('%d-%m-%y %H:%M:%S'), fn, feat_name, src))
            for key, val in sorted(kwargs.items()):
                op.write(' {}:{}'.format(key,val))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s  %(levelname)s:%(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default=DEFAULT_TOKEN_JAR, help='path to data')
    parser.add_argument('-op', '--outpath', type=str, default='/cip/lehre/CL4Spies/workspace/profiling_data/feats', help='path for saving numpy matrices')

    xsubparsers = parser.add_subparsers(title='Feature type', dest='feat_name')
    parser_xngram = xsubparsers.add_parser('ngram', help='Extract ngram features')
    parser_xngram.add_argument('-p', '--pos', action='store_true', help='use POS data !if standard directories aren\'t used the path to a file containing pickled POS data has to be supplied to --data')
    parser_xngram.add_argument('-n', '--ngram_range', type=str, default='1-1', help='ngram range in format \'n_min-n_max\'')
    parser_xngram.add_argument('-m', '--max_features', type=int, default=None, help='maximum feature count')
    parser_xngram.add_argument('-l', '--min_df', type=float, default=0.0, help='minimum document frequency')
    parser_xngram.add_argument('-u', '--max_df', type=float, default=1.0, help='maximum document frequency')
    parser_xstyle = xsubparsers.add_parser('style', help='Extract style features')
    parser_xread = xsubparsers.add_parser('read', help='Extract readability features')

    args = parser.parse_args()

    if args.feat_name == None:
        parser.print_help()
    else:
        if args.feat_name == 'ngram':
            kwargs = {'max_features': args.max_features,
                      'min_df': args.min_df,
                      'max_df': args.max_df,
                      'ngram_range': tuple(int(n) for n in args.ngram_range.split('-'))}
            extract_feats(data=args.data, feat_name=args.feat_name, outpath=args.outpath, pos=args.pos, **kwargs)
        else:
            extract_feats(data=args.data, feat_name=args.feat_name, outpath=args.outpath)

