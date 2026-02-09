#!/bin/bash
# Setup script that enables pip caching and installs dependencies

echo "Setting up virtual environment with pip caching enabled..."

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Enable pip caching (this will cache downloaded packages for future use)
echo "Enabling pip cache..."
pip install --upgrade pip

# Install packages with cache enabled
echo "Installing dependencies (this may take a while on first run)..."
echo "Future venvs can reuse cached packages!"
pip install --cache-dir ~/.cache/pip torch torchvision numpy matplotlib tqdm pillow

echo ""
echo "✓ Setup complete!"
echo "To activate the venv, run: source venv/bin/activate"
echo ""
echo "Note: Packages are now cached. Future venvs can reuse them if you use:"
echo "  pip install --cache-dir ~/.cache/pip <package>"





