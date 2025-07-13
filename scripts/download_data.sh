#!/bin/bash

# Downloads MusicNet dataset to local 'data' directory in the project's root level.
# Reference: https://github.com/jthickstun/pytorch_musicnet/blob/master/musicnet.py
# Usage: ./scripts/download_data.sh

set -e  # Exit on any error

# Configuration
DATA_DIR="../data/raw"
MUSICNET_URL="https://zenodo.org/records/5120004/files/musicnet.tar.gz"
MUSICNET_METADATA_URL="https://zenodo.org/records/5120004/files/musicnet_metadata.csv"

echo "Downloading MusicNet dataset..."

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download the file
wget -O "$DATA_DIR/musicnet.tar.gz" "$MUSICNET_URL"
echo "Download complete! File saved to: $DATA_DIR/musicnet.tar.gz"

# Download the metadata file
wget -O "$DATA_DIR/musicnet_metadata.csv" "$MUSICNET_METADATA_URL"
echo "Metadata downloaded to: $DATA_DIR/musicnet_metadata.csv"

# Extract the .tar.gz file
echo "Extracting dataset..."
tar -xzf "$DATA_DIR/musicnet.tar.gz" -C "$DATA_DIR"
echo "Extraction complete!"