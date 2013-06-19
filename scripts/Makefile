# This makefile should be located inside a model directory
# where all the necessary files will be created.
SOURCE_LANG?=en
TARGET_LANG?=es
MAX_TRIES?=10
TMX_FILE?=training.tmx

wordscore.csv: phrase-table.0-0.gz
	yalign-phrasetable-csv $^ wordscore.csv

phrase-table.0-0.gz: training_data
	touch $@

training_samples.csv: training_data
	yalign-generate-samples $^ training_samples.csv

tuscore.pickle: training_samples.csv wordscore.csv
	yalign-tuscore $^ tuscore.pickle

metadata.json: optimizing_data tuscore.pickle
	yalign-make-metadata -s $(SOURCE_LANG) -t $(TARGET_LANG) -n $(MAX_TRIES)  $^

# 70% of the corpus
training_data:
	yalign-transform $(TMX_FILE)

# 20% of the corpus
optimizing_data:
	yalign-transform $(TMX_FILE)

# 10% of the corpus
test_data:
	yalign-transform $(TMX_FILE)

create-model: metadata.json
