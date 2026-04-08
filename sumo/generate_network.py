"""
generate_network.py
Generates sumo/network.net.xml by calling netconvert.
Run this once before launching the simulation:

    python generate_network.py

Requires SUMO to be installed (netconvert must be in PATH or SUMO_HOME set).
"""
import os
import subprocess
import sys

SUMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo")
NET_FILE = os.path.join(SUMO_DIR, "network.net.xml")


def find_netconvert():
    """Find netconvert executable."""
    # 1. Try PATH directly
    import shutil
    nc = shutil.which("netconvert")
    if nc:
        return nc
    # 2. Try SUMO_HOME env var
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        candidates = [
            os.path.join(sumo_home, "bin", "netconvert"),
            os.path.join(sumo_home, "bin", "netconvert.exe"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    # 3. Common Windows install paths
    for root in [r"C:\Program Files (x86)\Eclipse\Sumo", r"C:\Sumo", r"C:\Program Files\Sumo"]:
        candidate = os.path.join(root, "bin", "netconvert.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def generate():
    nc = find_netconvert()
    if not nc:
        print("ERROR: netconvert not found.")
        print("Make sure SUMO is installed and either:")
        print("  - netconvert is in your PATH, or")
        print("  - SUMO_HOME environment variable is set")
        sys.exit(1)

    print(f"Using netconvert: {nc}")

    # Delete old hand-written network if present
    if os.path.exists(NET_FILE):
        os.remove(NET_FILE)
        print(f"Deleted old {NET_FILE}")

    cmd = [
        nc,
        "--node-files",       os.path.join(SUMO_DIR, "nodes.nod.xml"),
        "--edge-files",       os.path.join(SUMO_DIR, "edges.edg.xml"),
        "--connection-files", os.path.join(SUMO_DIR, "connections.con.xml"),
        "--tllogic-files",    os.path.join(SUMO_DIR, "tllogic.tll.xml"),
        "--output-file",      NET_FILE,
        "--no-turnarounds",   "true",
        "--junctions.corner-detail", "5",
        "--tls.default-type", "static",
        "--sidewalks.guess",  "false",
        "--crossings.guess",  "false",
        "--no-warnings",
    ]

    print("Running netconvert...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("netconvert FAILED:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    if not os.path.exists(NET_FILE):
        print("ERROR: network.net.xml was not created.")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print(f"SUCCESS: {NET_FILE} generated.")

    # Create output dir
    out_dir = os.path.join(SUMO_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory ready: {out_dir}")
    print("\nNext step: sumo -c sumo/config.sumocfg")


if __name__ == "__main__":
    generate()
