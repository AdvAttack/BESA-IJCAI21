# BESA-IJCAI21

This repository contains the implementations of the IJCAI-21 paper: BESA: BERT-based Simulated Annealing for Adversarial Text Attacks.

## Usage

Our code can be easily tested by the command line like: 
<pre><code>textattack attack --recipe bert-sa --model bert-base-uncased-imdb --num-examples 1000
</code></pre>
This will attack BERT model on IMDB dataset using our BESA.

## Citation

When using this code, or the ideas of BESA, please cite the following paper
<pre><code>@inproceedings{yang2021besa,
  title={BESA: BERT-based Simulated Annealing for Adversarial Text Attacks},
  author={Xinghao Yang and Weifeng Liu and Dacheng Tao and Wei Liu},
  booktitle={Proceedings of the 30th International Joint Conference on Artificial Intelligence},
  year={2021}
}
</code></pre>
