# -*- coding: utf-8 -*-
"""
Module providing tokenizers for various languages.
"""

from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer  # FIXME: It's an overkill
import re

###
### Common section
###
default_tokenizer = WordPunctTokenizer()

basic_macros = {
    "AN1": "[a-z0-9]",
    "AN2": "[a-z0-9\\._]",
    "AN3": r"[a-z0-9-_\.~!*'();:@&=+$,/?%#\[\]]"
}
macros = {
    "USERNAME": "{AN1}{AN2}*",
    "HOSTNAME": "{AN1}{AN2}*",
    "HOSTNAME2": r"{AN1}{AN2}*\.{AN2}*",
    "HOSTNAME3": r"{AN1}{AN2}*(:[0-9]{{1,5}})?",
    "HOSTNAME4": r"www\.{AN1}{AN2}*\.{AN2}*(:[0-9]{{1,5}})?",
    "SCHEME": "mailto:|((http|https|ftp|ftps|ssh|git|news)://)",
}
macros = {k: "(" + v.format(**basic_macros) + ")"
                                                for k, v in macros.items()}
macros.update(basic_macros)

eyes = ":;8xX>="
noses = [""] + list("-o")
mouths = list("DP/") + ["}}", "{{", "\\[", "\\]", "\\(", "\\)", "\\|"]
smileys = [x + y + z for x in eyes for y in noses for z in mouths]

HEADER = [
    "([01]?[0-9]|2[0-4]):[0-5]?[0-9](:[0-5]?[0-9])?",  # Time of day
    "''|``",                                           # Quotation
    "{USERNAME}@{HOSTNAME2}",                          # Typical email
    "{SCHEME}({USERNAME}@)?{HOSTNAME3}(/{AN3}*)?",     # URI
    "{HOSTNAME4}",                                     # Typical URL
]

FOOTER = [
    "\w+&\w+",                                         # And words
    "\w+",                                             # Normal words
    "|".join(smileys),                                 # Smileys
    "[()/\[\]\\.,;:\-\"'`~?]|\\.\\.\\.",               # Punctuation marks
    "\S+",                                             # Anything else
]
languages = {}


def get_tokenizer(language):
    """
    Get a tokenizer for a two character language code.
    """
    tokenizer = default_tokenizer
    if language in languages:
        regex = languages[language]
        regex = [x.format(**macros) for x in regex]
        regex = u"|".join(regex)
        tokenizer = RegexpTokenizer(regex, flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.I)
    return tokenizer


###
### English
###

english_contractions = [
 "ain't",
 "aren't",
 "can't",
 "can't've",
 "'cause",
 "could've",
 "couldn't",
 "couldn't've",
 "didn't",
 "doesn't",
 "don't",
 "hadn't",
 "hadn't've",
 "hasn't",
 "haven't",
 "he'd",
 "he'd've",
 "he'll",
 "he'll've",
 "he's",
 "how'd",
 "how'd'y",
 "how'll",
 "how's",
 "I'd",
 "I'd've",
 "I'll",
 "I'll've",
 "I'm",
 "I've",
 "isn't",
 "it'd",
 "it'd've",
 "it'll",
 "it'll've",
 "it's",
 "let's",
 "ma'am",
 "might've",
 "mightn't",
 "mightn't've",
 "must've",
 "mustn't",
 "mustn't've",
 "needn't",
 "o'clock",
 "oughtn't",
 "oughtn't've",
 "shan't",
 "shan't've",
 "she'd",
 "she'd've",
 "she'll",
 "she'll've",
 "she's",
 "should've",
 "shouldn't",
 "shouldn't've",
 "so's",
 "that's",
 "there'd",
 "there's",
 "they'd",
 "they'll",
 "they'll've",
 "they're",
 "they've",
 "to've",
 "wasn't",
 "we'd",
 "we'll",
 "we'll've",
 "we're",
 "we've",
 "weren't",
 "what'll",
 "what'll've",
 "what're",
 "what's",
 "what've",
 "when's",
 "when've",
 "where'd",
 "where's",
 "where've",
 "who'll",
 "who'll've",
 "who's",
 "who've",
 "why's",
 "will've",
 "won't",
 "won't've",
 "would've",
 "wouldn't",
 "wouldn't've",
 "y'all",
 "y'all'd've",
 "y'all're",
 "y'all've",
 "you'd",
 "you'd've",
 "you'll",
 "you'll've",
 "you're",
 "you've"]

languages["en"] = HEADER + [
    "[01]?[0-9][-/.][0123]?[0-9][-/.][0-9]{{2,4}}",    # Date mm/dd/yyyy
    "|".join(english_contractions),                    # Common contractions
    "'s",                                              # Possesive
    "\w+([_-]\w+)+",                                   # Normal words+compounds
] + FOOTER


###
### Spanish
###

languages["es"] = HEADER + [
    "[0123]?[0-9][-/.][01]?[0-9][-/.][0-9]{{2,4}}",    # Date dd/mm/yyyy
    u"¡¿",                                             # Extra punctuation mark
] + FOOTER


###
### Portuguese
###

languages["pt"] = HEADER + [
    "[0123]?[0-9][-/.][01]?[0-9][-/.][0-9]{{2,4}}",    # Date dd/mm/yyyy
    u"¡¿",                                             # Extra punctuation mark
    "\w+(-\w+)+",                                      # Compound words
] + FOOTER
