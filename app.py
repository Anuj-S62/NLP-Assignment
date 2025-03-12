# NER using pretrained BERT model 
# Imports
from transformers import BertForTokenClassification, BertTokenizer
# FastAPI
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import uvicorn



# Constants
max_length = 150
bert_model = 'bert-base-uncased'
finetuned_bert_model = 'fine_tuned_ner_model'

bert_model = BertForTokenClassification.from_pretrained(finetuned_bert_model)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
bert_label_dict = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

# FastAPI app
app = FastAPI()

def preprocess(text):
    if(isinstance(text, list)):
        text = ' '.join(text)
    tokens = []
    tokens = bert_tokenizer.tokenize(text)
    tokens = [t.lower() for t in tokens]
    return tokens

def predict(text):
    tokens = preprocess(text)
    input_ids = bert_tokenizer.encode(tokens, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predictions = predictions[:, 1:]
    predicted_label_ids = predictions[0].numpy()
    predicted_labels = [label_dict[id] for id in predicted_label_ids]
    return list(zip(tokens, predicted_labels))

def annotate_prediction(predictions):
    # annotate the text with the predicted labels
    annotated_text = ""
    for token, label in predictions:
        if(label=='O'):
            annotated_text += f"{token} "
        else:
            annotated_text += f"{token} ({label}) "
    return annotated_text


# API Endpoints
@app.get("/")
def read_root():
    return {"message ": "Welcome to NER API"}

# /predict endpoint
class Text(BaseModel):
    text: str

@app.post("/predict")
def predict_ner(data: Text):
    text = data.text
    predictions = predict(text)
    return {"text": text, "predictions": predictions, "annotated_text": annotate_prediction(predictions)}

# run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


