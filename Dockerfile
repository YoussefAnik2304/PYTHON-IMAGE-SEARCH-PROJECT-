# Use the Anaconda3 base image
FROM continuumio/anaconda3:2023.09-0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create the conda environment
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["conda", "run", "-n", "myenv", "python", "app.py"]