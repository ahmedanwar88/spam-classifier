# spam-classifier
This repository includes finetuning and deployment of a spam classifier model.

Model development is based on [Fine-Tuning-BERT repo](https://github.com/prateekjoshi565/Fine-Tuning-BERT/tree/master).

## Installation
The requirements can be installed using the following command:
```bash
pip3 install requirements.txt
```

## Training
The training notebook can be found [here](notebooks\Spam_Classifier.ipynb)

## Inference
The model can be tested using a text input as follows:
```bash
python3 inference.py --weights path_to_weights --text "input text"
```

## Application
The FastAPI can be run as follows:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

In the provided example, the app reads the model weights under ckpts folder with the name 'saved_weights_accuracy_distilbert.pt'.