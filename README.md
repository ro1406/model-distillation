# Model Distillation
Code and workshop material for model distillation workshop at Google x Nvidia event in Dubai - Jan 2026




## Deploy on Google Cloud Run

Command to run on Google Cloud Console
```bash
gcloud run deploy distilled-classifier \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```


Sample Curl request:
```bash
  curl -X POST https://YOUR-SERVICE-URL.run.app/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The workshop was absolutely fantastic and I learned a lot!"}'
```