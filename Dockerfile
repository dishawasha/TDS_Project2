# Use the version we know exists and works
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

# Switch to root to perform reconfiguration
USER root

# FIX: Remove the existing user with UID 1000 so we can create our own
RUN userdel -r $(id -un 1000) && \
    useradd -m -u 1000 user

# Switch to the new 'user'
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy files with correct ownership
COPY --chown=user . $HOME/app

# Install dependencies
# Note: This will now install playwright==1.48.0 from your requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the correct port for Hugging Face
EXPOSE 7860

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
