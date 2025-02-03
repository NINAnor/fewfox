#!/bin/bash

TARGET_DIR="/home/benjamin.cretois/data/nigens"
mkdir -p "$TARGET_DIR"
wget -q -O "$TARGET_DIR/NIGENS.zip" "https://zenodo.org/record/2535878/files/NIGENS.zip?download=1" | \
    unzip -q "$TARGET_DIR/NIGENS.zip" -d "$TARGET_DIR" | \
    rm "$TARGET_DIR/NIGENS.zip"