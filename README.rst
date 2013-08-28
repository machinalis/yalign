About
=====

Yalign is a tool for extracting parallel sentences from comparable corpora.

`Statistical Machine Translation <http://en.wikipedia.org/wiki/Statistical_machine_translation>`_ relies on `parallel corpora <http://en.wikipedia.org/wiki/Parallel_text>`_ (eg.. `europarl <http://www.statmt.org/europarl/>`_) for training translation models. However these corpora are limited and take time to create. Yalign is designed to automate this process by finding sentences that are close translation matches from `comparable corpora <http://www.statmt.org/survey/Topic/ComparableCorpora>`_. This opens up avenues for harvesting parallel corpora from sources like translated documents and the web.

Implementation
==============

Yalign uses a sequence alignment algorithm (`Needleman-Wunch <http://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm>`_) to find an optimal alignment between the sentences in two given documents.
Only sentences that have a high probability of being translations are included in the final alignment.

During alignment the algorithm uses a classfier (`Support Vector Machine <http://en.wikipedia.org/wiki/Support_vector_machine>`_) to determine if two sentences are translations of each other.
This means that the quality of the alignment is dependent not only on the input but also on the quality of the trained classifier.
