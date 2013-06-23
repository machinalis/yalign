import os
import tempfile
from yalign import tuscore

def default_tuscore():
    base_path = os.path.dirname(os.path.abspath(__file__))
    word_scores = os.path.join(base_path, "data", "test_word_scores.csv")
    tus = os.path.join(base_path, "data", "test_tus.csv")
    _, classifier_filepath = tempfile.mkstemp()
    tuscore.train_and_save_classifier(tus, word_scores, classifier_filepath)
    return classifier_filepath
