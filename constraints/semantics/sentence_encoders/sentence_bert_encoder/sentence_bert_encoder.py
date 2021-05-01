from sentence_transformers import SentenceTransformer

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

class SentenceBertEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, metric="cosine", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        
        self.model = embedder.encode

    def encode(self, sentences):
        return self.model(sentences, convert_to_tensor=True)
