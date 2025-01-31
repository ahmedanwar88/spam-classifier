import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
import argparse
from transformers import AutoModel, BertTokenizerFast
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from models.distil_bert import BERT_Spam

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_spam(text, model, tokenizer):
    """
    Infers the spam status of a single input sentence.

    Args:
        text (str): The input sentence.
        model (nn.Module): The trained spam detection model.
        tokenizer (transformers.BertTokenizerFast): The BERT tokenizer.

    Returns:
        int: The predicted spam status (0 for not spam, 1 for spam).
    """

    # Tokenize the input sentence
    tokens_inference = tokenizer(text,
                                  max_length = 20,
                                  return_tensors='pt',
                                  pad_to_max_length=True,
                                  truncation=True)

    # Move inputs to the appropriate device (e.g., GPU)
    input_seq = tokens_inference['input_ids'].to(device)
    input_mask = tokens_inference['attention_mask'].to(device)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_seq, input_mask)
        outputs = nn.Softmax(dim=1)(outputs)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = outputs[0, predicted_class].item()
    return predicted_class, confidence

def main():

    parser = argparse.ArgumentParser(description="A script that takes a string input and prints it.")
    
    # Add a string argument
    parser.add_argument("--weights", type=str, required=True, help="Weights path")
    parser.add_argument("--text", type=str, required=True, help="Input string")

    # Parse arguments
    args = parser.parse_args()

    text = args.text
    weights_path = args.weights

    # import BERT-base pretrained model
    #bert = AutoModel.from_pretrained('bert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Load the BERT tokenizer
    #tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    #weights_path = ".\ckpts\saved_weights_accuracy_distilbert.pt"

    model = BERT_Spam(bert)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    model.to(device)
    model.eval()

    predicted_class, confidence = infer_spam(text, model, tokenizer)
    
    if predicted_class == 0:
        print("NOT spam")
    else:
        print("Spam")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()