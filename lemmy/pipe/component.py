# coding: utf8
"""A spaCy pipeline component."""
from spacy.tokens import Token
from spacy.symbols import PRON_LEMMA
from lemmy import Lemmatizer
from lemmy import load as load_lang


class LemmyPipelineComponent(object):
    """
    A pipeline component for spaCy.

    This wraps a trained lemmatizer for easy use with spaCy.
    """

    name = 'lemmy'

    def __init__(self, lemmatizer_obj):
        """Initialize a pipeline component instance."""
        self._internal = lemmatizer_obj

    def __call__(self, doc):
        """
        Apply the pipeline component to a `Doc` object.

        doc (Doc): The `Doc` returned by the previous pipeline component.
        RETURNS (Doc): The modified `Doc` object.
        """
        for token in doc:
            if token.pos_ == "PRON":
                lemma = PRON_LEMMA
            else:
                lemma = self._get_lemma(token)

            if not lemma:
                continue
            token.lemma_ = lemma
        return doc

    def _get_lemma(self, token):
        lemmas = self._internal.lemmatize(token.pos_, token.text)
        if len(lemmas) != 1:
            for l in lemmas:
                if token.orth_ == l:
                    continue
                return l
        return lemmas[0]


def load(lang="da"):
    lemmatizer = load_lang(lang)
    return LemmyPipelineComponent(lemmatizer)
