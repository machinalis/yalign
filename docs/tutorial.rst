Tutorial
========

In this tutorial you will create a yalign model to align english and spanish documents. This same process can be applied to create models in the languages that you require.

To follow along in this tutorial you will need to download and unpack the tutorial data:

    ::

        wget http://yalign.machinalis.com/tutorial.tar.gz
        tar -xvzf tutorial.tar.gz

You should now have tutorial folder with two files: **dictionary.csv** and **corpus.en-es**. These two files will be explained shortly. 

Let's go and create our model..

Create A Model
----------------

During alignment Yalign uses a trained classifier to classify if a sentence is a translation or not. This classifier is dependent on the languages of the sentences to be aligned. This means that we need to train a model for the two languages that we want to align.

Training a model requires two inputs:

**1. A dictionary (dictionary.csv)** 
  
    This is a csv file for the two languages to be aligned. Each line consists of a word, a translation and a translation probability. 
    An english to spanish dictionary (dictionary.csv) is provided in the tutorial folder.

    .. Note:: 

        **How can I create  other dictionaries?**
        
        If you have worked with phrase tables before you will recognise that this information can be gleaned from a phrase table of 1-Grams. For conveniance we have included a script, **yalign-phrasetable-csv**, to convert an existing phrase table to a csv file. 
  
**2. A parallel corpus (corpus.en-es)** 

    The second requirement for training is an existing parallel corpus. 
    
    An english to spanish corpus (corpora.en-es) is provided in the tutorial folder. 
    The format of this corpus is plaintext but tmx is also accepted.
    
    .. Note::

        **Where can I find other parallel corpora?**
        
        A good resource for corpora is the `opus <http://opus.lingfil.uu.se/>`_ site. 
    
So now that we have a dictionary and parallel corpus we are ready to use **yalign-train** to train our model.

In the tutorial folder: 

    Create a folder where the model will be saved:
    
        ::

            mkdir en-es
    
    Run **yalign-train**:
    
        ::

            yalign-train -a en -b es corpus.en-es dictionary.csv en-es

    Once the training is finished there should be two files in the model folder (en-es):

    - aligner.pickle: A pickle of the yalign model.
    - metadata.json: Some metadata and parameters for the model.  

We can now use this model to align english and spanish documents.

Align Two Documents
-------------------

To perform an alignment we use the **yalign-align** script. 

The script takes a model folder (en-es) and the documents (urls or plantext files) to align as input. The results of the alignment are then written to stdout.

Let's align an english and spanish wikipedia page using our new model.

::
        
    yalign-align -a en -b es en-es http://en.wikipedia.org/wiki/Antiparticle http://es.wikipedia.org/wiki/Antipart%C3%ADcula

And that's it. You have successfully created your first alignment model! 
These same steps can be followed to create models in other languages.
