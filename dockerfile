FROM python:3.12-slim

WORKDIR /app

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src ./src

EXPOSE 7860

# Start FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]