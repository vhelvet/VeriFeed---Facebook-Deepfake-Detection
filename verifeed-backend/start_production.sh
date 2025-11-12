#!/bin/bash
# ============================================
# VeriFeed Production Installation Script
# ============================================

set -e  # Exit on error

echo "=========================================="
echo "🚀 VeriFeed Production Installation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}❌ Do not run this script as root${NC}"
    exit 1
fi

# Check Python version
echo -e "\n${GREEN}➤${NC} Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo -e "${RED}❌ Python $REQUIRED_VERSION or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Install system dependencies
echo -e "\n${GREEN}➤${NC} Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    echo "Detected Debian/Ubuntu system"
    sudo apt-get update
    sudo apt-get install -y \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        cmake \
        libopencv-dev \
        libboost-all-dev \
        git
elif command -v yum &> /dev/null; then
    echo "Detected RHEL/CentOS system"
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        python3-devel \
        cmake \
        opencv-devel \
        boost-devel \
        git
else
    echo -e "${YELLOW}⚠️  Unknown package manager. Please install dependencies manually.${NC}"
fi

# Create virtual environment
echo -e "\n${GREEN}➤${NC} Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Activate virtual environment
echo -e "\n${GREEN}➤${NC} Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\n${GREEN}➤${NC} Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo -e "\n${GREEN}➤${NC} Installing Python dependencies..."
if [ -f "requirements_production.txt" ]; then
    pip install -r requirements_production.txt
else
    echo -e "${RED}❌ requirements_production.txt not found${NC}"
    exit 1
fi

# Verify installations
echo -e "\n${GREEN}➤${NC} Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo -e "${RED}❌ PyTorch installation failed${NC}"
python -c "import face_recognition; print('Face Recognition: OK')" || echo -e "${RED}❌ Face Recognition installation failed${NC}"
python -c "import waitress; print('Waitress: OK')" || echo -e "${RED}❌ Waitress installation failed${NC}"
python -c "from flask_limiter import Limiter; print('Flask-Limiter: OK')" || echo -e "${RED}❌ Flask-Limiter installation failed${NC}"

# Check CUDA availability
echo -e "\n${GREEN}➤${NC} Checking CUDA availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\n${GREEN}➤${NC} Creating .env file..."
    cat > .env << 'EOF'
# ============================================
# PRODUCTION SECURITY CONFIGURATION
# ============================================

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
FLASK_SECRET_KEY=CHANGE_THIS_TO_RANDOM_VALUE

# DoS Prevention
MAX_CONTENT_MB=100
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=20
RATE_LIMIT_PER_HOUR=200
RATE_LIMIT_PER_DAY=1000
REQUEST_TIMEOUT=60
MAX_FRAMES_INPUT=600

# JWT Secret Key
JWT_SECRET_KEY=CHANGE_THIS_TO_RANDOM_VALUE

# Authentication
API_KEY=CHANGE_THIS_TO_RANDOM_VALUE
VERIFEED_AUTH_TOKEN=CHANGE_THIS_TO_RANDOM_VALUE

# CORS (comma-separated)
ALLOWED_ORIGINS=chrome-extension://iljbbfgejddphakhekbonjioflbodjoh,http://localhost

# Admin
ADMIN_API_KEY=CHANGE_THIS_TO_RANDOM_VALUE

# Model Security
ALLOW_MODEL_RELOAD=false
MODELS_DIR=models
EOF

    echo -e "${YELLOW}⚠️  .env file created with default values${NC}"
    echo -e "${YELLOW}⚠️  CRITICAL: Update all keys before running in production!${NC}"
    
    # Generate secure keys
    echo -e "\n${GREEN}➤${NC} Generating secure keys..."
    echo "Add these to your .env file:"
    echo ""
    echo "FLASK_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')"
    echo "JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')"
    echo "API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    echo "ADMIN_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    echo ""
else
    echo -e "${YELLOW}⚠️  .env file already exists. Skipping creation...${NC}"
fi

# Set file permissions
echo -e "\n${GREEN}➤${NC} Setting file permissions..."
chmod 600 .env
chmod 755 app8_production_secured.py

# Check if models directory exists
if [ ! -d "models" ]; then
    echo -e "\n${YELLOW}⚠️  models/ directory not found. Creating...${NC}"
    mkdir -p models
    echo -e "${YELLOW}⚠️  Please add your model file to models/ directory${NC}"
else
    echo -e "${GREEN}✓${NC} models/ directory found"
fi

# Test run (optional)
echo -e "\n${GREEN}➤${NC} Would you like to test the installation? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "\n${GREEN}➤${NC} Running health check..."
    export FLASK_DEBUG=True
    timeout 5 python app8_production_secured.py &
    SERVER_PID=$!
    sleep 3
    
    curl -s http://localhost:5000/health && echo -e "\n${GREEN}✓${NC} Server is running!" || echo -e "\n${RED}❌ Server health check failed${NC}"
    
    kill $SERVER_PID 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update .env file with your secure keys"
echo "2. Place your model file in models/ directory"
echo "3. Run: source venv/bin/activate"
echo "4. Run: python app8_production_secured.py"
echo ""
echo "For production deployment, see DEPLOYMENT_GUIDE.md"
echo ""