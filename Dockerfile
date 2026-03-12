# TaskTastic API – Python base (pdfplumber) + Node.js for API
FROM python:3.11-slim-bookworm

# Install Node.js 20 and system deps for pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    poppler-utils \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python PDF dependencies first (Python image = reliable pip)
RUN pip install --no-cache-dir "pdfplumber>=0.10.0"

# Install Node dependencies
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev 2>/dev/null || npm install --omit=dev

# Copy app
COPY . .

# Render sets PORT; default 3001
EXPOSE 3001
ENV PORT=3001

CMD ["node", "api/server.js"]
