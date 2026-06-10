FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    gliclass \
    torch \
    huggingface_hub \
    wandb \
    pandas \
    flask

COPY run_model.py .

CMD ["python", "run_model.py"]
