FROM python:3.12-slim

WORKDIR /app

# Install app dependencies only (no ML stack)
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy the full project (code only — data comes via volume/commit)
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
