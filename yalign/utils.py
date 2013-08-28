"""
Module for miscellaneous functions.
"""
import random
from string import letters
from lxml.builder import ElementMaker
from lxml import etree

def host_and_page(url):
    url = url.split('//')[1]
    parts = url.split('/')
    host = parts[0]
    page = "/".join(parts[1:])
    return host, '/' + page


def read_from_url(url):
    import httplib
    host, page = host_and_page(url)
    conn = httplib.HTTPConnection(host)
    conn.request("GET", page)
    response = conn.getresponse()
    return response.read()


def write_tmx(output_file, sentence_pairs, language_a, language_b):

    maker = ElementMaker()
    token = "".join(random.sample(letters * 3, 50))
    token_a = "".join(random.sample(letters * 3, 50))
    token_b = "".join(random.sample(letters * 3, 50))

    f = open(output_file, "w")

    header = maker.header(srclang=language_a,
                          segtype="sentence",
                          creationtool="MTrans",
                          datatype="PlainText")
    f.write("<?xml version=\"1.0\" ?>\n")
    f.write("<!DOCTYPE tmx SYSTEM \"tmx14.dtd\">\n")
    f.write("<tmx version=\"1.4\">\n")
    f.write(etree.tostring(header, encoding="utf-8"))
    f.write("\n<body>\n")

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
        f.write(tu_text)
    f.write("</body>\n</tmx>")
