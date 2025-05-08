#!/bin/bash
# Script to set up a Docker environment for PyForestLidar

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code
COPY . .

# Set the entrypoint to the CLI
ENTRYPOINT ["python", "cli.py"]

# Default command (can be overridden)
CMD ["--help"]
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.20.0
geopandas>=0.10.0
joblib>=1.1.0
matplotlib>=3.5.0
psutil>=5.9.0
click>=8.0.0
tqdm>=4.62.0
pandas>=1.3.0
scikit-learn>=1.0.0
# Add other dependencies as needed
EOF

# Create default config
mkdir -p config
cat > config/default_config.json << 'EOF'
{
    "kwargs": {
        "srs": "EPSG:2154",
        "hag": false,
        "crop_poly": false,
        "outlier": null,
        "smrf": false,
        "only_vegetation": false
    }
}
EOF

# Build Docker image
echo "Building Docker image..."
docker build -t pyforestlidar .

# Create run script
cat > run_pyforestlidar.sh << 'EOF'
#!/bin/bash
# Run the PyForestLidar Docker container

# Pass all arguments to the Docker container
docker run -it --rm \
    -v $(pwd):/app \
    -v "$PWD/data":/data \
    -v "$PWD/output":/output \
    --memory=8g \
    --cpus=4 \
    pyforestlidar "$@"
EOF

chmod +x run_pyforestlidar.sh

# Create data and output directories
mkdir -p data output

echo "Docker setup complete!"
echo "To run PyForestLidar, use: ./run_pyforestlidar.sh [COMMAND] [ARGS]"
echo "Example: ./run_pyforestlidar.sh process /data/input /output/result --jobs 4"
