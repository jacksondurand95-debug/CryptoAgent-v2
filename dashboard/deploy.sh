#!/bin/bash
# Deploy dashboard API route to JRD-Crypto-Dashboard
# Run from the Mound Mac Mini after git pull

DASHBOARD_DIR="$HOME/JRD-Crypto-Dashboard"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$DASHBOARD_DIR" ]; then
  echo "ERROR: Dashboard not found at $DASHBOARD_DIR"
  exit 1
fi

# Copy API route
cp "$SCRIPT_DIR/api_data_route.js" "$DASHBOARD_DIR/app/api/data/route.js"
echo "Deployed api/data/route.js"

echo "Done. Restart dashboard: cd $DASHBOARD_DIR && npx next dev -p 3099"
