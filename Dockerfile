# Gunakan base Python
FROM python:3.10-slim

# Install dependencies sistem untuk OpenCV (Wajib dijalankan sebagai root sebelum pindah user)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Buat user non-root (wajib di Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy requirements dan install dependencies Python
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file aplikasi (termasuk folder artifacts)
COPY --chown=user . $HOME/app

# Jalankan Streamlit di port 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]