# This makefile should be located inside a model directory
# where all the necesary files will be created.
# This directory must contain training.tmx wich is a parallel
# corpus.

wordscore.pickle: phrase-table.0-0.gz
	#yalign-wordscore $^ wordscore.pickle
	touch $@

phrase-table.0-0.gz: training_data
	touch $@

tuscore.pickle: training_data wordscore.pickle
	#yalign-tuscore training_data wordscore.pickle tuscore.pickle
	touch $@

metadata.json: optimizing_data tuscore.pickle
	touch $@


# 70% of the corpus
training_data:
	yalign-transform `pwd`/training.tmx

# 20% of the corpus
optimizing_data:
	yalign-transform `pwd`/training.tmx

# 10% of the corpus
test_data:
	yalign-transform `pwd`/training.tmx
