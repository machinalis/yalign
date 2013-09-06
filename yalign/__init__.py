# -*- coding: utf-8 -*-

"""
Examples of how to use the yalign API:

.. code-block:: python

    # Load a model from model that was saved to a folder eg.. en-es:

    from yalign import YalignModel

    model = YalignModel.load('en-es')

    # Align text

    from yalign import text_to_document

    english_text = \"""Virginia's eyes filled with tears and she hid her head in her hands.
                       The Duke rose and kissed his wife lovingly.\"""

    spanish_text = \"""¿No tiene ningún lugar donde pueda dormir?
                       Los ojos de Virginia se llenaron de lágrimas y óculto su rostro entre los manos.\"""

    english_sentences = text_to_document(english_text, 'en')
    spanish_sentences = text_to_document(spanish_text, 'es')

    pairs = model.align(english_sentences, spanish_sentences)

    # Align html

    from yalign import html_to_document

    english_html = \"""<html><body><p>
                       Virginia's eyes filled with tears and she hid her head in her hands.
                       The Duke rose and kissed his wife lovingly."
                       </p></body></html>\"""

    spanish_html = \"""<html><body><p>
                       ¿No tiene ningún lugar donde pueda dormir?
                       Los ojos de Virginia se llenaron de lágrimas y óculto su rostro entre los manos.
                       </p></body></html>\"""

    english_sentences = html_to_document(english_html, 'en')
    spanish_sentences = html_to_document(spanish_html, 'es')

    pairs = model.align(english_sentences, spanish_sentences)

    # Align srt

    from yalign import srt_to_document

    english_srt = \"""1\\n00:00:49,160 --> 00:00:50,992\\n
                      <i>Virginia's eyes filled with tears and she hid her head in her hands.</i>\\n\\n
                      2\\n00:00:51,734 --> 00:00:53,577\\n
                      <i>The Duke rose and kissed his wife lovingly.</i>\"""

    spanish_srt = \"""1\\n00:00:49,160 --> 00:00:50,992\\n
                      <i>¿No tiene ningún lugar donde pueda dormir?</i>\\n\\n
                      2\\n00:00:51,734 --> 00:00:53,577\\n
                      <i>Los ojos de Virginia se llenaron de lágrimas y óculto su rostro entre los manos.</i>\"""

    english_sentences = srt_to_document(english_srt, 'en')
    spanish_sentences = srt_to_document(spanish_srt, 'es')

    pairs = model.align(english_sentences, spanish_sentences)

"""

from yalignmodel import YalignModel, basic_model
from input_conversion import html_to_document, text_to_document, srt_to_document

