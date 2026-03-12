# TaskTastic API – Node.js + Python (pdfplumber) for PDF import
FROM node:20-slim

# Install Python 3, pip, and system deps for pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    poppler-utils \
    libgl1-mesa-glx \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node dependencies
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev 2>/dev/null || npm install --omit=dev

# Install Python PDF dependencies (pdfplumber only; anthropic is optional, install separately if needed)
RUN python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir "pdfplumber>=0.10.0"

# Copy app
COPY . .

# Render sets PORT; default 3001
EXPOSE 3001
ENV PORT=3001

CMD ["node", "api/server.js"]
