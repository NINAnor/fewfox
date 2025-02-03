#!/bin/bash

# Set the target directory
TARGET_DIR="/home/benjamin.cretois/data/esc50"
mkdir -p "$TARGET_DIR"
wget -qO - "https://github.com/karoldvl/ESC-50/archive/master.tar.gz" | \
tar xzf - -C "$TARGET_DIR"