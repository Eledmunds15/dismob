#!/bin/bash

# Shell script to transfer sync 000_data with HPC
USER="mtp24ele"
HOST="stanage.shef.ac.uk"
SOURCE_DIR="/home/Ethan/Projects/atom_sims/dismob/"
DEST_DIR="/mnt/parscratch/users/mtp24ele/dismob/"

# Use rsync for efficient transfer
echo "Starting file transfer..."
rsync -avzP \
  --exclude="*/restarts/" \
  --exclude="*/dump/" \
  --exclude="00_envs/" \
  "$USER@$HOST:$DEST_DIR" "$SOURCE_DIR"

# Check if the transfer was successful
if [ $? -eq 0 ]; then
  echo "Transfer completed successfully! 🎉"
else
  echo "Transfer failed. 😔 Please check your connection and paths."
  exit 1
fi
