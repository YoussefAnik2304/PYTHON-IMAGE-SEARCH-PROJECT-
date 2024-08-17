# Use a Miniconda image as the base image
FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create a working directory
WORKDIR /app

# Copy the environment.yml file and install the dependencies
COPY environment.yml .

# Add necessary channels
RUN conda config --add channels conda-forge

# Install mamba and create the environment
RUN conda install mamba -n base -c conda-forge \
    && mamba env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "your-environment-name", "/bin/bash", "-c"]

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the Flask application
CMD ["conda", "run", "--no-capture-output", "-n", "your-environment-name", "flask", "run", "--host=0.0.0.0"]
