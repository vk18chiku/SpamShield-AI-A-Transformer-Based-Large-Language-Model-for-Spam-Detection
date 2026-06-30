FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
