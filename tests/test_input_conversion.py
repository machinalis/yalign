# -*- coding: utf-8 -*-


import unittest
from StringIO import StringIO
import os

from yalign.datatypes import Sentence
from yalign.input_conversion import documents_from_parallel_corpus, tokenize, \
                                    text_to_document, html_to_document


base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "data")


def reader(N):
    return StringIO('\n'.join([str(x) for x in xrange(N)]))


class TestDocumentsFromParallelCorpus(unittest.TestCase):

    def test_empty_input(self):
        self.assertEquals([], list(documents_from_parallel_corpus(StringIO())))


    def test_document_sizes_between_min_and_max(self):
        cnt, m, n = 0, 5, 10
        N = 10000
        for A, B in documents_from_parallel_corpus(reader(N * 2), m, n):
            self.assertTrue(m <= len(A) <= n)
            self.assertTrue(m <= len(B) <= n)
            cnt += 1
        self.assertTrue(N / n <= cnt <= N / m)

    def test_no_zero_as_min(self):
        for A, B in documents_from_parallel_corpus(reader(20), 0, 1):
            self.assertTrue(1 <= len(A) <= 1)
            self.assertTrue(1 <= len(B) <= 1)


class BaseTestTokenization(object):
    language = "en"
    text = ""
    expected = "" or []

    def test_expected_words_are_in_tokenization(self):
        words = tokenize(self.text, self.language)
        self.assertIsInstance(words, Sentence)
        if isinstance(self.expected, basestring):
            self.expected = self.expected.split()  # Yes, this is irony.
        for expected_word in self.expected:
            self.assertIn(expected_word, words)


class TestTokenizationEn1(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"The dog is hungry.The cat is evil."
    expected = u"dog hungry evil ."
    

class TestTokenizationEn2(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"It's 3:39 am, what do you want?"
    expected = u"It's 3:39 want ?"


class TestTokenizationEn3(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"Try with ssh://tom@hawk:2020 and tell me"
    expected = u"ssh://tom@hawk:2020"


class TestTokenizationEn4(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"Visit http://google.com"
    expected = u"http://google.com"


class TestTokenizationEn5(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"I'm ready for you all. Aren't you ready?"
    expected = u"all . Aren't"


class TestTokenizationEn6(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"Back to 10-23-1984 but not to 23/10/1984"
    expected = u"10-23-1984 23 10 1984"


class TestTokenizationEn7(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"User-friendliness is a must, use get_text."
    expected = u"User-friendliness must get_text ."


class TestTokenizationEn8(BaseTestTokenization, unittest.TestCase):
    language = "en"
    text = u"John's bar is cool, right :) XD? :panda"
    expected = u"John 's cool , :) XD ?"


class TestTokenizationEs1(BaseTestTokenization, unittest.TestCase):
    language = "es"
    text = u"Ahí hay un vaso, me lo podrías alcanzar?porfavor"
    expected = u"Ahí vaso , podrías alcanzar ? porfavor"


class TestTokenizationEs2(BaseTestTokenization, unittest.TestCase):
    language = "es"
    text = u"Me pueden 'contactar' en juancito@pepito.com"
    expected = u"' contactar juancito@pepito.com"


class TestTokenizationEs3(BaseTestTokenization, unittest.TestCase):
    language = "es"
    text = u"Visita www.com.com y gana premios (seguro)"
    expected = u"www.com.com ( seguro )"


class TestTokenizationEs3(BaseTestTokenization, unittest.TestCase):
    language = "pt"
    text = u"A expressão tornou-se bastante comum no internetês."
    expected = u"expressão tornou-se internetês"


class TestTokenizationEs3(BaseTestTokenization, unittest.TestCase):
    language = "pt"
    text = u"uma cantora e compositora norte-americana de R&B."
    expected = u"norte-americana R&B"


class BaseTestTextToDocument(object):
    language = "en"
    text = ""

    def test_contains_more_than_one_sentence(self):
        document = text_to_document(self.text, self.language)
        self.assertGreater(len(document), 1)
        for sentence in document:
            self.assertIsInstance(sentence, Sentence)
            for word in sentence:
                self.assertIsInstance(word, unicode)


class TestTextToDocumentEn(BaseTestTextToDocument, unittest.TestCase):
    language = "en"
    text = (u"The Bastard Operator From Hell (BOFH), a fictional character "
            u"created by Simon Travaglia, is a rogue system administrator who "
            u"takes out his anger on users (often referred to as lusers), "
            u"colleagues, bosses, and anyone else who pesters him with their "
            u"pitiful user created \"problems\".\n"
            u"The BOFH stories were originally posted in 1992 to Usenet by "
            u"Travaglia, with some being reprinted in Datamation. They were "
            u"published weekly from 1995 to 1999 in Network Week and since 2000"
            u" they have been published most weeks in The Register. They were "
            u"also published in PC Plus magazine for a short time, and several"
            u" books of the stories have also been released.")


class TestTextToDocumentEs(BaseTestTextToDocument, unittest.TestCase):
    language = "es"
    text = (u"El bombo posee un gran espectro dinámico y poder sonoro, y puede"
            u"golpearse con una gran variedad de mazas y baquetas para lograr "
            u"diversos matices o efectos. Además, el ataque —o modo de "
            u"iniciarse el sonido— y la resonancia —o vibración del "
            u"instrumento— influyen en su timbre. Las técnicas de ejecución "
            u"incluyen diferentes tipos de golpe como el legato o stacatto, "
            u"al igual que efectos como redobles, apagado, golpeos al unísono "
            u"o notas de gracia. Desde sus orígenes es además habitual su "
            u"empleo junto a los platillos.")


class TestTextToDocumentPt(BaseTestTextToDocument, unittest.TestCase):
    language = "pt"
    text = (u"O casamento tinha a oposição dos governos do Reino Unido e dos "
            u"territórios autônomos da Commonwealth. Objeções religiosas, "
            u"jurídicas, políticas e morais foram levantadas. Como monarca "
            u"britânico, Eduardo era o chefe nominal da Igreja da Inglaterra, "
            u"que não permitia que pessoas divorciadas se casassem novamente "
            u"se seus ex-cônjuges ainda estivessem vivos; por isso, "
            u"acreditava-se que Eduardo não poderia casar-se com Wallis Simpson"
            u" e permanecer no trono. Simpson era considerada política e "
            u"socialmente inadequada como consorte devido aos seus dois "
            u"casamentos fracassados​​. O Establishment entendia que ela era "
            u"movida pelo amor ao dinheiro ou à posição e não por amor ao rei."
            u" Apesar da oposição, Eduardo declarou que amava Wallis e que "
            u"pretendia casar-se com ela, com ou sem a aprovação "
            u"governamental.")


class TestHtmlToDocument(unittest.TestCase):
    def test_generates_something(self):
        text = open(os.path.join(data_path, "index.html")).read()
        document = html_to_document(text, "en")
        self.assertGreater(len(document), 1)
        for sentence in document:
            self.assertIsInstance(sentence, Sentence)
            for word in sentence:
                self.assertIsInstance(word, unicode)


if __name__ == "__main__":
    unittest.main()
