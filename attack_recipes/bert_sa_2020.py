from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.semantics.sentence_encoders import SentenceBertEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import SimulatedAnnealnig
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapMaskedLM


def BertSA2020(model):
    """

    BERT-based Simulated Annealing for Word-level Text Adversarial Attacks

    """

    # We only consider the top K=50 synonyms from the MLM predictions.
    #
    transformation = WordSwapMaskedLM(method="bert-sa", max_candidates=50)
    #
    # Don't modify the same word twice or stopwords.
    #
    constraints = [RepeatModification(), StopwordModification()]
    constraints.append(PartOfSpeech(allow_verb_noun_swap=True))

    # "To ensure semantic similarity on introducing perturbations in the input
    # text, we use Universal Sentence Encoder (USE) (Cer et al., 2018)-based 
    # sentence similarity scorer."    
    use_constraint = UniversalSentenceEncoder(
        threshold=0.5,
        metric="cosine",
        compare_against_original=True,
        # window_size=15,
        skip_text_shorter_than_window=True,
    )
    constraints.append(use_constraint)
    #
    # Goal s untargeted classification.
    #
    goal_function = UntargetedClassification(model)
    
    
    search_method = SimulatedAnnealnig("best-substitution")
    
    return Attack(goal_function, constraints, transformation, search_method)
