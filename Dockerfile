# Use the Miniconda base image
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file and app code
COPY environment.yml .
COPY . .

# Create the environment (replace 'myenv' with your actual environment name)
RUN conda env create -f environment.yml

# Make sure the environment is activated:
SHELL ["conda", "run", "-n", "resume-analyzer", "/bin/bash", "-c"]

# Install streamlit as a fallback if not in env (optional)
RUN conda run -n resume-analyzer pip install streamlit

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["conda", "run", "--no-capture-output", "-n", "resume-analyzer", "streamlit", "run", "app.py"]
