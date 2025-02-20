# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Set a specific version for pip
RUN pip install --upgrade pip==21.1.2 && pip install -r requirements.txt

# Clean up cache to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Use a non-root user for security
RUN useradd -m appuser
USER appuser

# Copy the rest of the project files
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false"]
