from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from models.distil_bert import BERT_Spam

# Initialize the FastAPI app
app = FastAPI()

# import BERT-base pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')


# Load the BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

weights_path = ".\ckpts\saved_weights_accuracy_distilbert.pt"

model = BERT_Spam(bert)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define the input schema using Pydantic
class TextInput(BaseModel):
    text: str

# Define the inference function
def infer_spam(text: str):
    """
    Infers the spam status of a single input sentence.

    Args:
        text (str): The input sentence.

    Returns:
        dict: A dictionary containing the predicted class and confidence.
    """
    # Tokenize the input sentence
    tokens_inference = tokenizer(text,
                                  max_length=20,
                                  return_tensors='pt',
                                  pad_to_max_length=True,
                                  truncation=True)
    # Move inputs to the appropriate device
    input_seq = tokens_inference['input_ids'].to(device)
    input_mask = tokens_inference['attention_mask'].to(device)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_seq, input_mask)
        outputs = nn.Softmax(dim=1)(outputs)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = outputs[0, predicted_class].item()

    return {"predicted_class": predicted_class, "confidence": confidence}

# Define the API route
@app.post("/predict/")
async def predict(input: TextInput):
    try:
        result = infer_spam(input.text)
        response = {
            "text": input.text,
            "predicted_class": "Spam" if result["predicted_class"] == 1 else "Not Spam",
            "confidence": round(result["confidence"], 2)
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))