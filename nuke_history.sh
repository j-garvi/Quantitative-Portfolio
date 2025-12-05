#!/bin/bash

# NUCLEAR OPTION SCRIPT
# This script rewrites the git history to remove the username 'jesusg'
# from 'systematic-strategies/dual-momentum/garvi_jesus_momentum.ipynb'.

FILE_PATH="systematic-strategies/dual-momentum/garvi_jesus_momentum.ipynb"

echo "WARNING: This will rewrite your git history. Make sure you have a backup."
echo "Processing..."

# Use git filter-branch to rewrite history
# We use 'sed' to replace the personal path with a generic one in all past commits.
git filter-branch --force --tree-filter "
if [ -f \"$FILE_PATH\" ]; then
    sed -i 's|/Users/jesusg|/Users/user|g' \"$FILE_PATH\"
fi
" --tag-name-filter cat -- --all

echo "------------------------------------------------"
echo "History rewrite complete."
echo "1. Verify the history is clean."
echo "2. Run 'git push --force' to apply changes to your remote repository."
echo "------------------------------------------------"
