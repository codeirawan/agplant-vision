# Use a slim version of the Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that your app will run on
EXPOSE 8080

# Define the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
