import re
from random import choice, randint
from itertools import islice
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from scramble import shuffle, remove


def read_lines(parallel_corpus, n):
    """
    Read n lines of source and target in parallel corpus.
    Returns tuple of src and tgt lines.
    Each line consists of an index and sentence.
    The index can be used to match the sentences as
    matching sentences have the same index number.
    Eg.. returns [(1, 'hello'),(2, 'goodbye')],[(1, 'hola'), (2, 'adios')]
    """
    N = n * 2
    xs = list(x.decode('utf-8').strip()
              for x in islice(parallel_corpus, N))
    src = enumerate(xs[0:N:2])
    tgt = enumerate(xs[1:N:2])
    return list(src), list(tgt)


def generate_documents(parallel_corpus):
    """
    Returns two scrambled documents derived from the parallel_corpus.
    Each line is a tuple of an index and sentence. The index can be used
    to match pairs of sentences together.
    """
    n = randint(10, 30)
    src, tgt = read_lines(parallel_corpus, n)
    while src:
        src = scramble(src)
        tgt = scramble(tgt)
        yield src, tgt
        n = randint(10, 30)
        src, tgt = read_lines(parallel_corpus, n)


def samples(source):
    """
    Generate aligned and non-aligned training samples.
    Sample output for target t and source s is:
    {aligned: 0 or 1}, {s doc length}, {s index}, s, {t doc length}, {t index}, t
    """
    for src, tgt in generate_documents(source):
        for sample in aligned_samples(src, tgt):
            yield sample
        for sample in non_aligned_samples(src, tgt):
            yield sample


def aligned_samples(src, tgt):
    for idx, pair in enumerate(aligned_sentences(src, tgt)):
        a, b = pair
        yield 1, len(src), src.index(a), a[1], len(tgt), tgt.index(b), b[1]


def non_aligned_samples(src, tgt):
    N = max(len(src), len(tgt))
    for idx in xrange(N):
        a, b = choice(src), choice(tgt)
        if not a[0] == b[0]:
            yield 0, len(src), src.index(a), a[1], len(tgt), tgt.index(b), b[1]


def alignments(src, tgt):
    """
    Returns list of src and tgt alignments.
    Rg.. [(0, 0), (1, 2), (2, 1)]
    """
    for a, b in aligned_sentences(src, tgt):
        yield src.index(a), tgt.index(b)


def aligned_sentences(src, tgt):
    src_dict = dict(src)
    tgt_dict = dict(tgt)
    indexes = list(src_dict.keys())
    indexes.sort()
    for idx in indexes:
        a = src_dict.get(idx, None)
        b = tgt_dict.get(idx, None)
        if a and b:
            yield (idx, a), (idx, b)


def scramble(xs):
    ys = []
    n = 0
    r = randint(5, 10)
    x = xs[0:r]
    while x:
        n = n + r
        #remove(x, randint(0, 2))
        shuffle(x, randint(1, 3))
        ys += x
        r = randint(5, 10)
        x = xs[n:n + r]
    return ys

BAD_CHARS_PATTERN = re.compile('(\n|\t)+')


def text_to_corpus(text):
    return [re.sub(BAD_CHARS_PATTERN, ' ', x.strip()) for x in sent_tokenize(text)]


def html_to_text(reader):
    soup = BeautifulSoup(reader)
    text = soup.body.get_text()
    return text_to_corpus(text)

