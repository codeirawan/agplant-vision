FROM python:3.9-alpine

# Install dependencies
RUN apk add --no-cache \
    build-base \
    libjpeg-turbo-dev \
    zlib-dev

# Set up the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port and run the application
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
