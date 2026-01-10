#!/bin/bash
set -e

# Install dependencies if node_modules missing
if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
fi

# Compile TypeScript
echo "Compiling TypeScript..."
npx tsc --project tsconfig.json
echo "Compilation complete."
