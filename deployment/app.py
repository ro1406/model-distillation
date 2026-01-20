from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI()


MODEL_PATH = "./model/"
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Label Mapping
id2label = {0: "NEGATIVE", 1: "POSITIVE"}


@app.post("/predict")
async def predict(request: Request):
    # Get JSON body
    payload = await request.json()
    text = payload.get("text", "")

    if not text:
        return {"error": "No text provided"}

    # Tokenize
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
    print("Inputs keys after tokenization:", inputs.keys())

    # Remove token_type_ids if present (not needed for DistilBERT)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    inputs.pop("token_type_ids", None)
    print("Inputs keys before inference:", inputs.keys())

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    prediction = id2label.get(predicted_class_id, "UNKNOWN")
    confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

    return {"text": text, "prediction": prediction, "confidence": round(confidence, 4)}


# Health check endpoint for Cloud Run
@app.get("/")
def home():
    return {"status": "Model is running!"}
