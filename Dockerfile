FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential

WORKDIR /app

ENV BLIS_ARCH=generic

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
