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

**Note:**
Make sure your Dockerfile path logic is bulletproof. If you are inside llm-distillation-workshop/deployment and run `gcloud run deploy ... with --source .`:

- The context is the current folder.
- The Dockerfile says COPY ./model /app/model.
    This means the folder `llm-distillation-workshop/deployment/model` MUST exist and contain the weights before running the deploy command.