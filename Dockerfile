# TaskTastic API – Node.js + Python (pdfplumber) for PDF import
FROM node:20-slim

# Install Python 3 and pip for pdf_to_csv.py (pdfplumber)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node dependencies
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev 2>/dev/null || npm install --omit=dev

# Install Python PDF dependencies
COPY requirements-pdf.txt ./
RUN pip3 install --no-cache-dir -r requirements-pdf.txt

# Copy app
COPY . .

# Render sets PORT; default 3001
EXPOSE 3001
ENV PORT=3001

CMD ["node", "api/server.js"]
