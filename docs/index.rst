.. Yalign documentation master file, created by
   sphinx-quickstart on Mon Jun 17 10:46:02 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Yalign
======

Yalign is a tool for extracting parallel sentences from comparable corpora.

`Statistical Machine Translation <http://en.wikipedia.org/wiki/Statistical_machine_translation>`_ relies on `parallel corpora <http://en.wikipedia.org/wiki/Parallel_text>`_ (eg.. `europarl <http://www.statmt.org/europarl/>`_) for training translation models. However these corpora are limited and take time to create. Yalign is designed to automate this process by finding sentences that are close translation matches from `comparable corpora <http://www.statmt.org/survey/Topic/ComparableCorpora>`_. This opens up avenues for harvesting parallel corpora from sources like translated documents and the web.

Installation
============

Yalign requires that you install `scikit-learn <http://scikit-learn.org/stable/install.html>`_.

After that you can install Yalign from PyPi via pip:

::

    sudo pip install yalign

Usage
=====

Firstly we need to download and unpack the english to spanish model.

::

    wget http://yalign.machinalis.com/models/0.1/en-es.tar.gz
    tar -xvzf en-es.tar.gz 

Now we can use the **yalign-align** script along with the english to spanish model to align two web pages.

::

    yalign-align en-es http://en.wikipedia.org/wiki/Antiparticle http://es.wikipedia.org/wiki/Antipart%C3%ADcula

Yalign is not limited to any one language pair. By creating your own models you can align any two languages. 

**Contents:**

.. toctree::
   :maxdepth: 2

   installation
   tutorial
   implementation
   reference 


Yalign is a `Machinalis <http://www.machinalis.com>`_ project.
You can view our other open source contributions `here <https://github.com/machinalis/>`_.

**The Yalign Team:**

| Laura Alonso Alemany
| Elías Andrawos
| Rafael Carrascosa
| Gonzalo García Berrotarán
| Andrew Vine

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

