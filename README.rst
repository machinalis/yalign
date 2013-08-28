**Introduction**

Yalign is a tool for extracting parallel sentences from comparable corpora.

Statistical Machine translation relies on parallel corpora (eg.. Europarl) for training translation models.
However these corpora are limited and take time to create. Yalign is designed to automate this process by
finding sentences that are close translation matches from comparable corpora. This opens up avenues for 
harvesting parallel corpora from sources like translated documents and the web.

**Implementation**


Yalign uses a sequence alignment algorithm (Needleman-Wunch) to find an optimal alignment between the sentences in two given documents.
Only sentences that have a high probability of being translations are included in the final alignment.

During alignment the algorithm uses a classfier (SVM) to determine if two sentences are translations of each other.
This means that the quality of the alignment is dependent not only on the input but also on the quality of the trained classifier.
