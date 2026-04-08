#!/usr/bin/env python3
"""
Generate SUMO network using netconvert for guaranteed compatibility.
Creates nodes and edges files, then uses netconvert to build the network.
"""

import subprocess
import os

def create_nodes_file():
    """Create nodes.nod.xml"""
    nodes = [
        ("W0", -250, 0, "dead_end"),
        ("E2", 1250, 0, "dead_end"),
        ("N0", 0, 250, "dead_end"),
        ("S0", 0, -250, "dead_end"),
        ("N1", 500, 250, "dead_end"),
        ("S1", 500, -250, "dead_end"),
        ("N2", 1000, 250, "dead_end"),
        ("S2", 1000, -250, "dead_end"),
        ("J0", 0, 0, "traffic_light"),
        ("J1", 500, 0, "traffic_light"),
        ("J2", 1000, 0, "traffic_light"),
    ]
    
    nod_xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    nod_xml.append('<nodes version="1.26" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    nod_xml.append('       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">')
    
    for node_id, x, y, node_type in nodes:
        nod_xml.append(f'  <node id="{node_id}" x="{x:.2f}" y="{y:.2f}" type="{node_type}"/>')
    
    nod_xml.append('</nodes>')
    
    with open("sumo/nodes.nod.xml", "w", encoding="utf-8") as f:
        f.write("\n".join(nod_xml))
    print("Created nodes.nod.xml")

def create_edges_file():
    """Create edges.edg.xml"""
    edges = [
        ("W0J0", "W0", "J0", 13.89, 2, 9),
        ("J0J1", "J0", "J1", 13.89, 2, 9),
        ("J1J0", "J1", "J0", 13.89, 2, 9),
        ("J1J2", "J1", "J2", 13.89, 2, 9),
        ("J2J1", "J2", "J1", 13.89, 2, 9),
        ("J2E2", "J2", "E2", 13.89, 2, 9),
        ("N0J0", "N0", "J0", 11.11, 2, 7),
        ("J0N0", "J0", "N0", 11.11, 2, 7),
        ("S0J0", "S0", "J0", 11.11, 2, 7),
        ("J0S0", "J0", "S0", 11.11, 2, 7),
        ("N1J1", "N1", "J1", 11.11, 2, 7),
        ("J1N1", "J1", "N1", 11.11, 2, 7),
        ("S1J1", "S1", "J1", 11.11, 2, 7),
        ("J1S1", "J1", "S1", 11.11, 2, 7),
        ("N2J2", "N2", "J2", 11.11, 2, 7),
        ("J2N2", "J2", "N2", 11.11, 2, 7),
        ("S2J2", "S2", "J2", 11.11, 2, 7),
        ("J2S2", "J2", "S2", 11.11, 2, 7),
    ]
    
    edg_xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    edg_xml.append('<edges version="1.26" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    edg_xml.append('       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">')
    
    for edge_id, from_node, to_node, speed, lanes, priority in edges:
        edg_xml.append(f'  <edge id="{edge_id}" from="{from_node}" to="{to_node}" '
                      f'priority="{priority}" numLanes="{lanes}" speed="{speed:.2f}"/>')
    
    edg_xml.append('</edges>')
    
    with open("sumo/edges.edg.xml", "w", encoding="utf-8") as f:
        f.write("\n".join(edg_xml))
    print("Created edges.edg.xml")

def create_tllogic_file():
    """Create tllogic.tll.xml"""
    tll_xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    tll_xml.append('<tlLogics version="1.26" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    tll_xml.append('          xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/tllogic_file.xsd">')
    
    tl_configs = [
        ("TL0", "0", ["GGggGGggrrrr", "yyyyyyyyrrrr", "rrrrrrrrGGGG", "rrrrrrrryyyy"]),
        ("TL1", "36", ["GGggGGggrrrr", "yyyyyyyyrrrr", "rrrrrrrrGGGG", "rrrrrrrryyyy"]),
        ("TL2", "72", ["GGggrrrr", "yyyyrrrr", "rrrrGGGG", "rrrryyyy"]),
    ]
    
    for tl_id, offset, states in tl_configs:
        tll_xml.append(f'  <tlLogic id="{tl_id}" type="static" programID="coordinated" offset="{offset}">')
        phases = [
            ("35", states[0], "EW_green"),
            ("5", states[1], "EW_yellow"),
            ("30", states[2], "NS_green"),
            ("5", states[3], "NS_yellow"),
        ]
        for duration, state, name in phases:
            tll_xml.append(f'    <phase duration="{duration}" state="{state}" name="{name}"/>')
        tll_xml.append('  </tlLogic>')
    
    tll_xml.append('</tlLogics>')
    
    with open("sumo/tllogic.tll.xml", "w", encoding="utf-8") as f:
        f.write("\n".join(tll_xml))
    print("Created tllogic.tll.xml")

def build_network_with_netconvert():
    """Use netconvert to build the network"""
    os.chdir("sumo")
    
    cmd = [
        "netconvert",
        "-n", "nodes.nod.xml",
        "-e", "edges.edg.xml",
        "-t", "tllogic.tll.xml",
        "-o", "network_generated.net.xml",
        "--no-internal-links"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Verify file was created
    if os.path.exists("network_generated.net.xml"):
        # Move to final name
        os.replace("network_generated.net.xml", "network.net.xml")
        print("Network file generated successfully!")
        return True
    return False

if __name__ == "__main__":
    import os
    os.makedirs("sumo", exist_ok=True)
    
    create_nodes_file()
    create_edges_file()
    create_tllogic_file()
    
    if build_network_with_netconvert():
        print("\n✓ SUMO network created successfully")
    else:
        print("\n✗ Failed to create network")
