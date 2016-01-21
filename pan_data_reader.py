#!/usr/bin/python3

import logging
import argparse
import pickle
import xml.etree.ElementTree as et
import re
from html import unescape
from html.parser import HTMLParser
from lxml import etree as let


GENDER_NAMES = ['male', 'female']
AGE_NAMES = ['10s', '20s', '30s']
DEFAULT_TEST_DATA = 'texts-en.xml'
DEFAULT_PICKLE_JAR = 'texts-en.pkl'
URL_REGEX = re.compile(r"""
\b
(                                       # Capture 1: entire matched URL
  (?:
    [a-z][\w-]+:                        # URL protocol and colon
    (?:
      /{1,3}                            # 1-3 slashes
      |                                 #   or
      [a-z0-9%]                         # Single letter or digit or '%'
                                        # (Trying not to match e.g. "URI::Escape")
    )
    |                                   #   or
    www\d{0,3}[.]                       # "www.", "www1.", "www2." … "www999."
    |                                   #   or
    [a-z0-9.\-]+[.][a-z]{2,4}/          # looks like domain name followed by a slash
  )
  (?:                                   # One or more:
    [^\s()<>]+                          # Run of non-space, non-()<>
    |                                   #   or
    \(([^\s()<>]+|(\([^\s()<>]+\)))*\)  # balanced parens, up to 2 levels
  )+
  (?:                                   # End with:
    \(([^\s()<>]+|(\([^\s()<>]+\)))*\)  # balanced parens, up to 2 levels
    |                                   #   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]      # not a space or one of these punct char
  )
)""", re.VERBOSE | re.IGNORECASE)  # https://gist.github.com/gruber/249502


class TagProcessor(HTMLParser):
    digested = []

    def handle_data(self, d):
        self.digested.append(d)

    def get_nutrients(self):
        return ''.join(self.digested)

    def do_it(self):
        self.digested = []


def replace_url(match):
    #URL replacement depending on file extensions
    img_exts = ('.png', '.jpg', '.jpeg')
    if match.group(0).endswith(img_exts):
        return '[URL-IMG]'
    else:
        return '[URL]'


def _preprocess(s:str):
    parser = TagProcessor()
    unesc = unescape(s)
    parser.do_it()
    parser.feed(unesc.strip())
    text = parser.get_nutrients()
    text = URL_REGEX.sub(replace_url, text)
    return text


def read_pan_data(fn:str, gender_names=GENDER_NAMES, age_names=AGE_NAMES):
    #Read blog data from xml in PAN13 format.
    gender_dic = {v:i for i,v in enumerate(gender_names)}
    age_dic = {v:i for i,v in enumerate(age_names)}

    texts = []
    genders = []
    ages = []
    logging.info('Read PAN13 format data from {}'.format(fn))

    elements = let.iterparse(fn, events=["end"])
    n_authors = 0
    for event, el in elements:
        if el.tag == 'conversation':
            t = el.text or ''
            texts.append(_preprocess(t))
            el.clear()
        elif el.tag == 'author':
            n_authors += 1
            gender = gender_dic[el.attrib['gender']]
            age = age_dic[el.attrib['age_group']]
            genders.extend([gender] * (len(texts) - len(genders)))
            ages.extend([age] * (len(texts) - len(ages)))
        elif el.tag == 'file':
            del el.getparent()[0]
        else:
            continue
    logging.info('{} authors'.format(n_authors))
    return texts, genders, ages


def read_data(fn:str=DEFAULT_PICKLE_JAR):
    logging.info('Read preprocessed PAN13 data from {}'.format(fn))
    fh = open(fn, 'rb')
    texts, genders, ages = pickle.load(fh)
    return texts, genders, ages


class TagFinder(HTMLParser):
    #Parser class for _process_tags
    tagcollection = set()

    def handle_starttag(self, tag, attrs):
        self.tagcollection.add(tag)

    def write_tags(self, fn:str):
        xml_log = open(fn, 'w')
        for tag in self.tagcollection:
            xml_log.write('{}\n'.format(tag))
        xml_log.close()


def _process_tags(raw:list):
    #Data exploration function for finding and logging remaining xml/html tags
    # stops a 20% for some reason
    logging.info('Processing remaining html tags')
    parser = TagFinder()
    d_infothresholds = {int((i/100.0*len(raw))):"%i%%"%(i) for i in range(0, 101, 5)}
    for i, text in enumerate(raw):
        unesc = unescape(text)
        parser.feed(unesc)

        if i in d_infothresholds.keys():
            parser.write_tags('xml_tags.log')
            logging.info('{} of documents processed'.format(d_infothresholds[i]))
    xml_log.close()

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s  %(levelname)s:%(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', '-r', type=str, default=DEFAULT_TEST_DATA, help='path to original PAN xml data')
    parser.add_argument('--out', '-o', type=argparse.FileType('wb'), default=None, help='destination file for pickled data')
    args = parser.parse_args()

    out = args.out or open(DEFAULT_PICKLE_JAR, 'wb')
    texts, genders, ages = read_pan_iter_data(args.raw)
    logging.info('Pickling results to {}'.format(out.name))
    pickle.dump((texts, genders, ages), out)
