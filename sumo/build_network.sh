#!/usr/bin/env bash
# =============================================================================
#  build_network.sh  -  Jaam Ctrl
#  Generates network.net.xml from source files using netconvert.
#
#  Run from the project root:
#    bash sumo/build_network.sh          (Linux / macOS / Git Bash on Windows)
#    OR use build_network.ps1 on Windows PowerShell
#
#  Requires: SUMO installed and in PATH (netconvert command available)
#  After running: network.net.xml will be created in sumo/
# =============================================================================

set -e

SUMO_DIR="$(dirname "$0")"

echo "================================================="
echo " Jaam Ctrl – Building SUMO network with netconvert"
echo "================================================="

netconvert \
  --node-files        "$SUMO_DIR/nodes.nod.xml" \
  --edge-files        "$SUMO_DIR/edges.edg.xml" \
  --connection-files  "$SUMO_DIR/connections.con.xml" \
  --tllogic-files     "$SUMO_DIR/tllogic.tll.xml" \
  --output-file       "$SUMO_DIR/network.net.xml" \
  --no-turnarounds    true \
  --junctions.corner-detail  5 \
  --junctions.limit-turn-speed  5.5 \
  --tls.default-type  static \
  --tls.cycle.time    75 \
  --sidewalks.guess   false \
  --crossings.guess   false \
  --geometry.remove   false \
  --verbose           true

echo ""
echo "SUCCESS: network.net.xml written to $SUMO_DIR/"
echo "Next: sumo -c sumo/config.sumocfg"
