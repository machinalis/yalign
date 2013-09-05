"""
Module for miscellaneous functions.
"""
import random
from collections import defaultdict
from string import letters
from lxml.builder import ElementMaker
from lxml import etree


def host_and_page(url):
    """ Splits a `url` into the hostname and the rest of the url. """
    url = url.split('//')[1]
    parts = url.split('/')
    host = parts[0]
    page = "/".join(parts[1:])
    return host, '/' + page


def read_from_url(url):
    """ GET this `url` and read the response. """
    import httplib
    host, page = host_and_page(url)
    conn = httplib.HTTPConnection(host)
    conn.request("GET", page)
    response = conn.getresponse()
    return response.read()


def write_tmx(stream, sentence_pairs, language_a, language_b):
    """ Writes the SentencePair's out in tmx format, """
    maker = ElementMaker()
    token = "".join(random.sample(letters * 3, 50))
    token_a = "".join(random.sample(letters * 3, 50))
    token_b = "".join(random.sample(letters * 3, 50))

    header = maker.header(srclang=language_a,
                          segtype="sentence",
                          creationtool="MTrans",
                          datatype="PlainText")
    stream.write("<?xml version=\"1.0\" ?>\n")
    stream.write("<!DOCTYPE tmx SYSTEM \"tmx14.dtd\">\n")
    stream.write("<tmx version=\"1.4\">\n")
    stream.write(etree.tostring(header, encoding="utf-8"))
    stream.write("\n<body>\n")

    for sentence_a, sentence_b in sentence_pairs:
        src_tuv = maker.tuv({token: language_a}, maker.seg(token_a))
        tgt_tuv = maker.tuv({token: language_b}, maker.seg(token_b))

        tu = maker.tu(src_tuv, tgt_tuv)
        tu_text = etree.tostring(tu, encoding="utf-8",
                                 pretty_print=True)
        tu_text = tu_text.replace(token, "xml:lang")
        if sentence_a and sentence_b:
            tu_text = tu_text.replace(token_a, sentence_a.to_text())
            tu_text = tu_text.replace(token_b, sentence_b.to_text())
        stream.write(tu_text)
    stream.write("</body>\n</tmx>")


class CacheOfSizeOne(object):
    """ Function wrapper that provides caching. """
    f = None

    def __init__(self, f):
        self.f = f
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        if args != self.args or kwargs != self.kwargs:
            self.result = self.f(*args, **kwargs)
            self.args = args
            self.kwargs = kwargs
        return self.result

    def __getattr__(self, name):
        return getattr(self.f, name)


class Memoized(defaultdict):

    def __missing__(self, key):
        x = self.default_factory(key)
        self[key] = x
        return x
