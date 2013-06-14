from random import randint, shuffle as knuth_shuffle
import re
from StringIO import StringIO
from itertools import islice

PUNCTUATION = re.compile('[\?\!\.]')


def scramble(source=StringIO(),
             more=StringIO(),
             sentence_range=[5, 10],
             sentence_swap_range=[1, 2],
             sentence_insertion_range=[0, 2],
             sentence_removal_range=[0, 2],
             word_swap_range=[0, 2],
             word_insertion_range=[0, 2],
             word_removal_range=[0, 1]):
    """
    Input
        -source: A file object of the sentences separated by newlines.
        -more: A file object of sentences to be used as the source of insertions.
        -sentence_range: The range of sentence that scrambling will occur in.
        The following all act within the sentence range:
          -sentence_swap_range:       Range of sentence swaps.
          -sentence_insertions_range: Range of sentence insertions.
          -sentence_removal_range:    Range sentence removals.
        The following all act within one sentence:
          -word_swap_range:           Range of word swaps.
          -word_insertion_range:      Range of word insertions.
          -word_removal_range:        Range of word removals.
    """
    lines = next_group(source, randint(*sentence_range))
    while lines:
        more_lines = next_group(more, randint(*sentence_insertion_range))
        shuffle(lines, randint(*sentence_swap_range))
        lines = list(scramble_words(lines,
                                    lines_to_words(more_lines),
                                    word_swap_range,
                                    word_insertion_range,
                                    word_removal_range))
        remove(lines, randint(*sentence_removal_range))
        insert(lines, more_lines)
        for line in lines:
            yield line
        lines = next_group(source, randint(*sentence_range))


def remove_punctuation(words):
    match = re.match(PUNCTUATION, words[-1][-1])
    if match:
        words[-1] = words[-1][:-1]
        return match.group(0)
    return None


def decapitalize_first_word(words):
    if words and words[0][0]:
        words[0] = words[0][0].lower() + words[0][1:]


def capitalize_first_word(words):
    if words:
        words[0] = words[0].capitalize()


def insert_punctuation(words, punctuation):
    if words and punctuation:
        words[-1] += punctuation


def lines_to_words(lines):
    words = [word for line in lines for word in line.split()]
    knuth_shuffle(words)
    return words


def scramble_words(lines, words_to_insert, swap_range, insertion_range, removal_range):
    for line in lines:
        words = line.split()
        punctuation = remove_punctuation(words)
        decapitalize_first_word(words)
        shuffle(words, randint(*swap_range))
        remove(words, randint(*removal_range))
        offset = randint(*insertion_range)
        insert(words, words_to_insert[:offset])
        words_to_insert = words_to_insert[offset:]
        if words:
            capitalize_first_word(words)
            insert_punctuation(words, punctuation)
        else:
            words = line.split()
        yield ' '.join(words)


def next_group(reader, n):
    """
    Read a list of size n from the reader.
    """
    return list(islice(reader, 0, n))


def random_range(n):
    """
    Shuffled range of n elements.
    """
    indexes = range(n)
    knuth_shuffle(indexes)
    return indexes


def shuffle(xs, n):
    """
    Inplace shuffle of xs by performing n swaps.
    """
    if n > len(xs):
        n = len(xs)
    indexes = random_range(len(xs))
    for i in xrange(0, n):
        a, b = indexes[-1], indexes[i]
        xs[a], xs[b] = xs[b], xs[a]


def remove(xs, n):
    """
    Inplace random removal of n elements from xs.
    """
    N = min(n, len(xs))
    for _ in xrange(N):
        if not xs: break
        idx = randint(0, len(xs) - 1)
        xs.pop(idx)


def insert(xss, xs):
    """
    Inplace insertion of elements from xs into random positions in xss.
    """
    for x in xs:
        idx = randint(0, len(xss))
        xss.insert(idx, x)
