import numpy as np

from textattack.attack_results import FailedAttackResult, SkippedAttackResult

from . import CSVLogger, FileLogger, VisdomLogger, WeightsAndBiasesLogger
# Anonymous added new tools for semantic and grammar checking
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

# Anonymous added universal sentence encoder
import torch
import tensorflow_hub as hub


class AttackLogManager:
    """Logs the results of an attack to all attached loggers."""

    def __init__(self):
        self.loggers = []
        self.results = []

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def enable_wandb(self):
        self.loggers.append(WeightsAndBiasesLogger())

    def add_output_file(self, filename):
        self.loggers.append(FileLogger(filename=filename))

    def add_output_csv(self, filename, color_method):
        self.loggers.append(CSVLogger(filename=filename, color_method=color_method))

    def log_result(self, result):
        """Logs an ``AttackResult`` on each of `self.loggers`."""
        self.results.append(result)
        for logger in self.loggers:
            logger.log_attack_result(result)

    def log_results(self, results):
        """Logs an iterable of ``AttackResult`` objects on each of
        `self.loggers`."""
        for result in results:
            self.log_result(result)
        self.log_summary()

    def log_summary_rows(self, rows, title, window_id):
        for logger in self.loggers:
            logger.log_summary_rows(rows, title, window_id)

    def log_sep(self):
        for logger in self.loggers:
            logger.log_sep()

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_attack_details(self, attack_name, model_name):
        # @TODO log a more complete set of attack details
        attack_detail_rows = [
            ["Attack algorithm:", attack_name],
            ["Model:", model_name],
        ]
        self.log_summary_rows(attack_detail_rows, "Attack Details", "attack_details")

    def log_summary(self):
        total_attacks = len(self.results)
        if total_attacks == 0:
            return
        # Count things about attacks.
        all_num_words = np.zeros(len(self.results))
        perturbed_word_percentages = np.zeros(len(self.results))
        num_words_changed_until_success = np.zeros(
            2 ** 16
        )  # @ TODO: be smarter about this
        failed_attacks = 0
        skipped_attacks = 0
        successful_attacks = 0
        max_words_changed = 0
        # Anonymous added variables
        tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        use_encoder = hub.load(tfhub_url)
        all_use_scores = np.zeros(len(self.results))
        all_cos_scores = np.zeros(len(self.results))
        all_grammar_errors = np.zeros(len(self.results))
        adv_truelabel = []
        for i, result in enumerate(self.results):
            # Anonymous check semantic Sentence-BERT and grammar errors
            # original_text = result.original_result.attacked_text._text_input["text"]
            # perturbed_text = result.perturbed_result.attacked_text._text_input["text"]  # Anonymous added imdb, agnews, mr, yelp
            original_text = result.original_result.attacked_text._text_input["sentence"]
            perturbed_text = result.perturbed_result.attacked_text._text_input["sentence"]  # Anonymous added cola, SST-2
            # original_text = result.original_result.attacked_text._text_input["premise"] + result.original_result.attacked_text._text_input["hypothesis"]
            # perturbed_text = result.perturbed_result.attacked_text._text_input["premise"] + result.perturbed_result.attacked_text._text_input["hypothesis"]  # Anonymous added for SNLI
            # original_text = result.original_result.attacked_text._text_input["question"] + result.original_result.attacked_text._text_input["sentence"]
            # perturbed_text = result.perturbed_result.attacked_text._text_input["question"] + result.perturbed_result.attacked_text._text_input["sentence"]  # Anonymous added for QNLI
            '''
            # below two line collects adv and true labels used for transfer attack
            true_label = result.original_result.ground_truth_output
            adv_truelabel.append((perturbed_text, true_label))
            '''
            # calculate the universal sentence encoder
            use_ref_emb = use_encoder([original_text]).numpy()
            use_can_emb = use_encoder([perturbed_text]).numpy()
            use_ref_emb = torch.tensor(use_ref_emb)
            use_can_emb = torch.tensor(use_can_emb)
            all_use_scores[i] = util.pytorch_cos_sim(use_can_emb, use_ref_emb)[0]
            # calculate the semantic Sentence-Bert
            refs_embedding = embedder.encode(original_text, convert_to_tensor=True)
            cands_embedding = embedder.encode(perturbed_text, convert_to_tensor=True)
            all_cos_scores[i] = util.pytorch_cos_sim(cands_embedding, refs_embedding)[0]
            # check the number of grammar errors
            orig_error = tool.check(original_text)
            pertub_error = tool.check(perturbed_text)
            all_grammar_errors[i] = len(pertub_error) - len(orig_error)
            if all_grammar_errors[i] < 0:
                print("Fixed grammar errors for:", i, "origina:", len(orig_error), "perturb:", len(pertub_error))
                print("Original text:", original_text)
            
            all_num_words[i] = len(result.original_result.attacked_text.words)
            if isinstance(result, FailedAttackResult):
                failed_attacks += 1
                continue
            elif isinstance(result, SkippedAttackResult):
                skipped_attacks += 1
                continue
            else:
                successful_attacks += 1
            num_words_changed = len(
                result.original_result.attacked_text.all_words_diff(
                    result.perturbed_result.attacked_text
                )
            )
            num_words_changed_until_success[num_words_changed - 1] += 1
            max_words_changed = max(
                max_words_changed or num_words_changed, num_words_changed
            )
            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0
            perturbed_word_percentages[i] = perturbed_word_percentage

        # Original classifier success rate on these samples.
        original_accuracy = (total_attacks - skipped_attacks) * 100.0 / (total_attacks)
        original_accuracy = str(round(original_accuracy, 2)) + "%"

        # New classifier success rate on these samples.
        accuracy_under_attack = (failed_attacks) * 100.0 / (total_attacks)
        accuracy_under_attack = str(round(accuracy_under_attack, 2)) + "%"

        # Attack success rate.
        if successful_attacks + failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                successful_attacks * 100.0 / (successful_attacks + failed_attacks)
            )
        attack_success_rate = str(round(attack_success_rate, 2)) + "%"

        perturbed_word_percentages = perturbed_word_percentages[
            perturbed_word_percentages > 0
        ]
        average_perc_words_perturbed = perturbed_word_percentages.mean()
        average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"

        average_num_words = all_num_words.mean()
        average_num_words = str(round(average_num_words, 2))
        # Anonymous added variables
        average_use_score = all_use_scores.mean()
        average_use_score = str(round(average_use_score, 4))
        average_cos_score = all_cos_scores.mean()
        average_cos_score = str(round(average_cos_score, 4))
        average_grammar_error = all_grammar_errors.mean()
        average_grammar_error = str(round(average_grammar_error, 4))
        
        summary_table_rows = [
            ["Number of successful attacks:", str(successful_attacks)],
            ["Number of failed attacks:", str(failed_attacks)],
            ["Number of skipped attacks:", str(skipped_attacks)],
            ["Original accuracy:", original_accuracy],
            ["Accuracy under attack:", accuracy_under_attack],
            ["Attack success rate:", attack_success_rate],
            ["Average perturbed word %:", average_perc_words_perturbed],
            ["Average num. words per input:", average_num_words],
            ["USE score:", average_use_score],  # Anonymous added
            ["Sentence Bert score:", average_cos_score],  # Anonymous added
            ["Grammar errors:", average_grammar_error],  # Anonymous added
        ]

        num_queries = np.array(
            [
                r.num_queries
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        )
        avg_num_queries = num_queries.mean()
        avg_num_queries = str(round(avg_num_queries, 2))
        summary_table_rows.append(["Avg num queries:", avg_num_queries])
        self.log_summary_rows(
            summary_table_rows, "Attack Results", "attack_results_summary"
        )
        # Show histogram of words changed.
        numbins = max(max_words_changed, 10)
        for logger in self.loggers:
            logger.log_hist(
                num_words_changed_until_success[:numbins],
                numbins=numbins,
                title="Num Words Perturbed",
                window_id="num_words_perturbed",
            )
