# spam-classifier
This repository includes finetuning and deployment of a spam classifier model.

Model development is based on [Fine-Tuning-BERT repo](https://github.com/prateekjoshi565/Fine-Tuning-BERT/tree/master).

## Installation
The requirements can be installed using the following command:
```bash
pip3 install requirements.txt
```

## Inference
The model can be tested using a text input as follows:
```bash
python3 inference.py --weights path_to_weights --text "input text"
```
