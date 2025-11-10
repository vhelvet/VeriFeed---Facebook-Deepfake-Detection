🔐 Security Overview
Implemented Security Features
1. DoS Prevention

✅ Max Content Length: 100MB limit on request size
✅ Input Validation: Strict validation on frame counts and base64 data
✅ Rate Limiting: Configurable per-minute, per-hour, and per-day limits
✅ Request Timeout: 60-second timeout for long-running requests
✅ Frame Limits: Maximum 600 frames per request (configurable)
✅ Memory Protection: Decoded frame size limits (20MB per frame)

2. Authentication & Authorization

✅ API Key Authentication: Hashed comparison with caching
✅ JWT Token Support: 24-hour expiring tokens
✅ Admin-Only Endpoints: Separate admin key for sensitive operations
✅ CORS Restrictions: Limited to specified origins only

3. Code Hardening

✅ Production Mode: Debug mode disabled by default
✅ Secret Management: All secrets from environment variables
✅ Path Traversal Prevention: Validated model paths
✅ Generic Error Messages: No stack traces in production
✅ Security Headers: X-Frame-Options, X-Content-Type-Options, etc.

4. Infrastructure

✅ Waitress WSGI Server: Production-grade server
✅ Thread Pool: Configurable concurrent request handling
✅ Connection Limits: Maximum 1000 concurrent connections
✅ Server Identity Hidden: No version exposure


🛠️ Environment Setup
Prerequisites

Python 3.8+
pip
GPU (optional, but recommended for performance)
CUDA 11.8+ (if using GPU)

System Dependencies
Ubuntu/Debian:
bashsudo apt-get update
sudo apt-get install -y \
    python3-dev \
    build-essential \
    cmake \
    libopencv-dev \
    libboost-all-dev
CentOS/RHEL:
bashsudo yum groupinstall "Development Tools"
sudo yum install -y python3-devel cmake opencv-devel boost-devel

📦 Installation
1. Clone Repository
bashgit clone https://github.com/your-repo/verifeed-backend.git
cd verifeed-backend
2. Create Virtual Environment
bashpython3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
3. Install Dependencies
bashpip install --upgrade pip
pip install -r requirements_production.txt
4. Verify Installation
bashpython -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import face_recognition; print('Face Recognition: OK')"
python -c "import waitress; print('Waitress: OK')"

⚙️ Configuration
1. Create Environment File
Create a .env file in the project root:
bashcp .env.example .env
2. Configure Environment Variables
CRITICAL: Update these values before deployment!
properties# ============================================
# PRODUCTION SECURITY CONFIGURATION
# ============================================

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
FLASK_SECRET_KEY=YOUR_RANDOM_SECRET_KEY_HERE_CHANGE_THIS

# DoS Prevention
MAX_CONTENT_MB=100
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=20
RATE_LIMIT_PER_HOUR=200
RATE_LIMIT_PER_DAY=1000
REQUEST_TIMEOUT=60
MAX_FRAMES_INPUT=600

# JWT Secret Key (for signing tokens)
JWT_SECRET_KEY=YOUR_JWT_SECRET_KEY_HERE_CHANGE_THIS

# Authentication
API_KEY=YOUR_API_KEY_HERE_CHANGE_THIS
VERIFEED_AUTH_TOKEN=your_admin_authentication_token_here

# CORS (comma-separated)
ALLOWED_ORIGINS=chrome-extension://YOUR_EXTENSION_ID,https://yourdomain.com

# Admin
ADMIN_API_KEY=YOUR_ADMIN_KEY_HERE_CHANGE_THIS

# Model Security
ALLOW_MODEL_RELOAD=false
MODELS_DIR=models
3. Generate Secure Keys
Generate random secure keys:
bash# Generate API Key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT Secret
python -c "import secrets; print(secrets.token_hex(32))"

# Generate Admin Key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate Flask Secret
python -c "import secrets; print(secrets.token_hex(32))"
4. Set File Permissions
bashchmod 600 .env  # Only owner can read/write
chmod 755 app8_production_secured.py
chmod -R 755 models/

🚀 Running the Server
Development Mode (Testing Only)
bashexport FLASK_DEBUG=True
python app8_production_secured.py
Production Mode (Waitress)
bash# Ensure FLASK_DEBUG=False in .env
python app8_production_secured.py
Using systemd (Recommended)
Create /etc/systemd/system/verifeed.service:
ini[Unit]
Description=VeriFeed Deepfake Detection API
After=network.target

[Service]
Type=simple
User=verifeed
WorkingDirectory=/opt/verifeed-backend
Environment="PATH=/opt/verifeed-backend/venv/bin"
ExecStart=/opt/verifeed-backend/venv/bin/python app8_production_secured.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/verifeed-backend

[Install]
WantedBy=multi-user.target
Enable and start:
bashsudo systemctl daemon-reload
sudo systemctl enable verifeed
sudo systemctl start verifeed
sudo systemctl status verifeed
Using Docker (Alternative)
Dockerfile:
dockerfileFROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements_production.txt .
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 verifeed && chown -R verifeed:verifeed /app
USER verifeed

EXPOSE 5000

CMD ["python", "app8_production_secured.py"]
Build and run:
bashdocker build -t verifeed-api .
docker run -d \
  --name verifeed \
  -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/models:/app/models:ro \
  verifeed-api

🛡️ Security Best Practices
1. API Key Management

❌ Never commit API keys to version control
✅ Use environment variables or secret managers (AWS Secrets Manager, HashiCorp Vault)
✅ Rotate keys regularly (quarterly recommended)
✅ Use different keys for development and production

2. Network Security
bash# Allow only specific IPs (firewall)
sudo ufw allow from YOUR_CLIENT_IP to any port 5000
sudo ufw enable

# Or use nginx reverse proxy with rate limiting
3. HTTPS/TLS
Use nginx or Caddy as reverse proxy:
Nginx configuration:
nginxserver {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Rate limiting
        limit_req zone=api burst=10 nodelay;
    }
}

# Rate limit zone
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
4. Monitoring & Logging
bash# View logs
sudo journalctl -u verifeed -f

# Monitor resource usage
watch -n 1 'ps aux | grep python | grep -v grep'
htop

# Check open connections
netstat -an | grep :5000
5. Regular Updates
bash# Update dependencies
pip install --upgrade -r requirements_production.txt

# Check for security vulnerabilities
pip install safety
safety check

📊 Monitoring & Maintenance
Health Check
bashcurl http://localhost:5000/health
Expected response:
json{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "production_mode": true,
  "rate_limiting": true
}
Load Testing
bash# Install Apache Bench
sudo apt-get install apache2-utils

# Test endpoint
ab -n 100 -c 10 -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:5000/health
Performance Metrics
Monitor these metrics:

Request latency (p50, p95, p99)
Error rate
CPU/GPU utilization
Memory usage
Active connections

Backup Strategy
bash# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Backup configuration
cp .env .env.backup.$(date +%Y%m%d)

🐛 Troubleshooting
Common Issues
1. Model Not Loading
bash# Check model file exists
ls -lh models/

# Check permissions
ls -la models/model_acc_88.89_epoch25_20251108_095329.pt

# View detailed error
FLASK_DEBUG=True python app8_production_secured.py
2. Rate Limit Issues
bash# Check rate limit configuration
grep RATE_LIMIT .env

# Temporarily disable for testing
export RATE_LIMIT_ENABLED=false
3. Memory Issues
bash# Monitor memory
free -h
watch -n 1 free -h

# Reduce frame limits
export MAX_FRAMES_INPUT=300
export MAX_FRAMES_TO_PROCESS=30
4. CUDA Errors
bash# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
export CUDA_VISIBLE_DEVICES=-1

📞 Support
For issues or questions:

Check logs: sudo journalctl -u verifeed -n 100
GitHub Issues: https://github.com/your-repo/issues
Email: support@yourdomain.com

⚠️ SECURITY REMINDER:

Change all default keys before deployment
Enable HTTPS in production
Regularly update dependencies
Monitor logs for suspicious activity
Keep backups of models and configuration