# Gunakan base Python (bisa upgrade ke slim kalau mau ringan)
FROM python:3.10-slim

# Buat user non-root (wajib di HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file app (termasuk folder artifacts)
COPY --chown=user . $HOME/app

# Jalankan Streamlit di port 7860 (wajib!)
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
