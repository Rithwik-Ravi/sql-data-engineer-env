FROM python:3.10-slim

# Creates a non-root user with an explicit UID and GID required by Hugging Face Spaces
RUN groupadd -g 1000 user && \
    useradd -m -u 1000 -g 1000 user
USER 1000:1000

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create the working directory
WORKDIR $HOME/app

# Install dependencies first for Docker caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY --chown=user:user . .

# Expose the standard Hugging Face Space port
EXPOSE 7860

# Start the environment API server
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
