import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)

import math
import random
# from bert_score import score
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')


class SimulatedAnnealnig(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
    """

    def __init__(self, wir_method="unk"):
        self.wir_method = wir_method

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
        elif self.wir_method == "best-substitution":
            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text, original_text=initial_text, indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = np.array(delta_ps)
        elif self.wir_method == "weighted-saliency":
            
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            print("search_over:", search_over)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()
            
            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text, original_text=initial_text, indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)
        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = (-index_scores).argsort()

        search_over = False
        return index_order, search_over
      
    def _aim_function(self, initial_result, x):
        '''aim function: the smaller the better'''
        # compute victim model's prediction score
        output, _ = self.get_goal_results([x.attacked_text])
        true_score = 1 - output[0].score        
        # count the number of word substitutions
        if 'modified_indices' in x.attacked_text.attack_attrs:
            cost = len(x.attacked_text.attack_attrs['modified_indices'])
        else:
            cost = 0
        delta = 0.01  # tradeoff parameter
        y = true_score + delta*cost
        
        return y
    
    def _perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        
        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        
        # Starts Simulated Annealing
        T = 20  # initial temperature
        Tmin = 1  # lowest temperature
        k = 20  # internal iterations
        x = initial_result  # initialize x as original input text
        t = 0  # time
        while T >= Tmin:
            for i in range(k):
                # calculate y
                y = self._aim_function(initial_result, x)
                # randomly modify original x
                indice = t + random.randint(0, round(3*t))
                # indice = t + randomList[i]  # avoid repeated modification
                if indice >= len(index_order):
                    print("continue due to indice out of range")
                    continue
                moves = self.get_transformations(
                            x.attacked_text,
                            original_text=initial_result.attacked_text,
                            indices_to_modify=[index_order[indice]]
                            )
                # Skip words without candidates
                if len(moves) == 0:
                    continue
                xNew, _ = self.get_goal_results(moves)
                xNew_sorted = sorted(xNew, key=lambda x: -x.score)
                yNew = self._aim_function(initial_result, xNew_sorted[0])
                if yNew - y < 0:
                    x = xNew_sorted[0]
                else:
                    # metropolis principle
                    p = math.exp(-(yNew-y)/T)
                    r = np.random.uniform(low=0, high=1)
                    if r < p:
                        x = xNew_sorted[0]
                # If we succeeded, return the result.
                if x.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    return x
            t += 1
            T = 1000/(1+t)  # quick annealing function
        return x

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    def extra_repr_keys(self):
        return ["wir_method"]
