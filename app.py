from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Tuple
import torch
import uvicorn
import spacy
from transformers import BertForTokenClassification, BertTokenizer
import pickle

# Constants
MAX_LENGTH = 150
BERT_MODEL_NAME = "bert-base-uncased"
FINETUNED_BERT_MODEL = "fine_tuned_ner_model"
SPACY_MODEL_PATH = "spacy_ner_model.pkl"
BERT_LABEL_DICT = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}


class BERTNERModel:
    def __init__(self):
        self.model = BertForTokenClassification.from_pretrained(FINETUNED_BERT_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def preprocess(self, text: str) -> List[str]:
        if isinstance(text, list):
            text = " ".join(text)
        tokens = self.tokenizer.tokenize(text)
        tokens = [t.lower() for t in tokens]
        return tokens

    def predict(self, text: str) -> List[Tuple[str, str]]:
        tokens = self.preprocess(text)
        input_ids = self.tokenizer.encode(tokens, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        predictions = predictions[:, 1:]
        predicted_label_ids = predictions[0].numpy()
        predicted_labels = [BERT_LABEL_DICT[id] for id in predicted_label_ids]
        return list(zip(tokens, predicted_labels))

    def annotate_text(self, predictions: List[Tuple[str, str]]) -> str:
        annotated_text = ""
        for token, label in predictions:
            if label == 'O':
                annotated_text += f"{token} "
            else:
                annotated_text += f"{token} ({label}) "
        return annotated_text


class SpacyNERModel:
    def __init__(self):
        with open(SPACY_MODEL_PATH, 'rb') as input:
            spacy_trained_model = pickle.load(input)
        self.spacy_trained_model = spacy_trained_model

    
    def predict(self, text) -> List[Tuple[str, str]]:
        doc = self.spacy_trained_model(text)
        prediction = []
        for ent in doc.ents:
            prediction.append((ent.text, ent.label_))
        return prediction

    def annotate_text(self, predictions: List[Tuple[str, str]]) -> str:
        annotated_text = ""
        for token, label in predictions:
            if label == 'O':
                annotated_text += f"{token} "
            else:
                annotated_text += f"{token} ({label}) "
        return annotated_text


# Initialize FastAPI app
app = FastAPI()
bert_ner = BERTNERModel()
spacy_ner = SpacyNERModel()


class TextInput(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the NER API!"}


@app.post("/predict/bert")
def predict_ner_bert(data: TextInput):
    text = data.text
    predictions = bert_ner.predict(text)
    annotated_text = bert_ner.annotate_text(predictions)
    return {"text": text, "predictions": predictions, "annotated_text": annotated_text}


@app.post("/predict/spacy")
def predict_ner_spacy(data: TextInput):
    text = data.text
    predictions = spacy_ner.predict(text)
    annotated_text = spacy_ner.annotate_text(predictions)
    return {"text": text, "predictions": predictions, "annotated_text": annotated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
