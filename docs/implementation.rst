Implementation
==============

Yalign uses a sequence alignment algorithm (`Needleman-Wunch <http://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm>`_) to find an optimal alignment between the sentences in two given documents. Only sentences that have a high probability of being translations are included in the final alignment.
 
During alignment the algorithm uses a classfier (`Support Vector Machine <http://en.wikipedia.org/wiki/Support_vector_machine>`_) to determine if two sentences are translations of each other. This means that the quality of the alignment is dependent not only on the input but also on the quality of the trained classifier.
