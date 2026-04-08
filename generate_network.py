#!/usr/bin/env python3
"""
Generate a valid SUMO network file for Jaam Ctrl 3-intersection corridor.
This script creates a properly formatted network.net.xml with all required attributes.
"""

def generate_network():
    """Generate SUMO network XML with proper structure."""
    
    # Nodes definition: (id, x, y, type, tl_id)
    nodes = [
        ("W0", -250, 0, "dead_end", None),
        ("E2", 1250, 0, "dead_end", None),
        ("N0", 0, 250, "dead_end", None),
        ("S0", 0, -250, "dead_end", None),
        ("N1", 500, 250, "dead_end", None),
        ("S1", 500, -250, "dead_end", None),
        ("N2", 1000, 250, "dead_end", None),
        ("S2", 1000, -250, "dead_end", None),
        ("J0", 0, 0, "traffic_light", "TL0"),
        ("J1", 500, 0, "traffic_light", "TL1"),
        ("J2", 1000, 0, "traffic_light", "TL2"),
    ]
    
    # Edges: (id, from_node, to_node, edge_type, shape_coords)
    edges = [
        ("W0J0", "W0", "J0", "arterial", [(-250, 0), (0, 0)]),
        ("J0J1", "J0", "J1", "arterial", [(0, 0), (500, 0)]),
        ("J1J0", "J1", "J0", "arterial", [(500, 0), (0, 0)]),
        ("J1J2", "J1", "J2", "arterial", [(500, 0), (1000, 0)]),
        ("J2J1", "J2", "J1", "arterial", [(1000, 0), (500, 0)]),
        ("J2E2", "J2", "E2", "arterial", [(1000, 0), (1250, 0)]),
        ("N0J0", "N0", "J0", "crossroad", [(0, 250), (0, 0)]),
        ("J0N0", "J0", "N0", "crossroad", [(0, 0), (0, 250)]),
        ("S0J0", "S0", "J0", "crossroad", [(0, -250), (0, 0)]),
        ("J0S0", "J0", "S0", "crossroad", [(0, 0), (0, -250)]),
        ("N1J1", "N1", "J1", "crossroad", [(500, 250), (500, 0)]),
        ("J1N1", "J1", "N1", "crossroad", [(500, 0), (500, 250)]),
        ("S1J1", "S1", "J1", "crossroad", [(500, -250), (500, 0)]),
        ("J1S1", "J1", "S1", "crossroad", [(500, 0), (500, -250)]),
        ("N2J2", "N2", "J2", "crossroad", [(1000, 250), (1000, 0)]),
        ("J2N2", "J2", "N2", "crossroad", [(1000, 0), (1000, 250)]),
        ("S2J2", "S2", "J2", "crossroad", [(1000, -250), (1000, 0)]),
        ("J2S2", "J2", "S2", "crossroad", [(1000, 0), (1000, -250)]),
    ]
    
    # Connections: (from_edge, to_edge, from_lane, to_lane, dir, tl_id, link_index)
    # Connections arriving at J0 (TL0)
    connections = [
        ("W0J0", "J0J1", 0, 0, "s", "TL0", 0),
        ("W0J0", "J0J1", 1, 1, "s", "TL0", 1),
        ("W0J0", "J0S0", 0, 0, "r", "TL0", 2),
        ("W0J0", "J0N0", 1, 0, "l", "TL0", 3),
        ("J1J0", "J0N0", 0, 0, "l", "TL0", 4),
        ("J1J0", "J0N0", 1, 1, "l", "TL0", 5),
        ("J1J0", "J0S0", 0, 0, "r", "TL0", 6),
        ("J1J0", "J0J1", 1, 1, "s", "TL0", 7),
        ("N0J0", "J0J1", 0, 0, "r", "TL0", 8),
        ("N0J0", "J0S0", 1, 0, "s", "TL0", 9),
        ("S0J0", "J0J1", 0, 0, "l", "TL0", 10),
        ("S0J0", "J0N0", 1, 0, "s", "TL0", 11),
        # Connections arriving at J1 (TL1)
        ("J0J1", "J1J2", 0, 0, "s", "TL1", 0),
        ("J0J1", "J1J2", 1, 1, "s", "TL1", 1),
        ("J0J1", "J1S1", 0, 0, "r", "TL1", 2),
        ("J0J1", "J1N1", 1, 0, "l", "TL1", 3),
        ("J2J1", "J1J0", 0, 0, "s", "TL1", 4),
        ("J2J1", "J1J0", 1, 1, "s", "TL1", 5),
        ("J2J1", "J1S1", 0, 0, "l", "TL1", 6),
        ("J2J1", "J1N1", 1, 0, "r", "TL1", 7),
        ("N1J1", "J1J2", 0, 0, "r", "TL1", 8),
        ("N1J1", "J1J0", 1, 0, "s", "TL1", 9),
        ("S1J1", "J1J2", 0, 0, "l", "TL1", 10),
        ("S1J1", "J1J0", 1, 0, "s", "TL1", 11),
        # Connections arriving at J2 (TL2)
        ("J1J2", "J2E2", 0, 0, "s", "TL2", 0),
        ("J1J2", "J2E2", 1, 1, "s", "TL2", 1),
        ("J1J2", "J2S2", 0, 0, "r", "TL2", 2),
        ("J1J2", "J2N2", 1, 0, "l", "TL2", 3),
        ("N2J2", "J2E2", 0, 0, "r", "TL2", 4),
        ("N2J2", "J2J1", 1, 0, "s", "TL2", 5),
        ("S2J2", "J2E2", 0, 0, "l", "TL2", 6),
        ("S2J2", "J2J1", 1, 0, "s", "TL2", 7),
    ]
    
    # Generate XML
    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<net version="1.16" junctionCornerDetail="5" limitTurnSpeed="5.50"')
    xml.append('     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
    xml.append('     xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">')
    xml.append('')
    
    # Location
    xml.append('  <location netOffset="0.00,0.00" convBoundary="-250.00,-250.00,1250.00,250.00"')
    xml.append('            origBoundary="-250.00,-250.00,1250.00,250.00" projParameter="!"/>')
    xml.append('')
    
    # Types
    xml.append('  <type id="arterial"  priority="9" numLanes="2" speed="13.89" allow="all"/>')
    xml.append('  <type id="crossroad" priority="7" numLanes="2" speed="11.11" allow="all"/>')
    xml.append('')
    
    # Nodes
    for node_id, x, y, node_type, tl_id in nodes:
        tl_str = f' tl="{tl_id}"' if tl_id else ""
        xml.append(f'  <node id="{node_id}" x="{x:.2f}" y="{y:.2f}" type="{node_type}"{tl_str}/>')
    xml.append('')
    
    # Edges with lanes
    edge_types = {
        "arterial": ("13.89", 2),
        "crossroad": ("11.11", 2),
    }
    
    for edge_id, from_node, to_node, edge_type, shape_coords in edges:
        speed, num_lanes = edge_types[edge_type]
        shape_str = " ".join([f"{x:.2f},{y:.2f}" for x, y in shape_coords])
        xml.append(f'  <edge id="{edge_id}" from="{from_node}" to="{to_node}" type="{edge_type}" shape="{shape_str}">')
        
        # Lane shapes (offset by 1.75 m perpendicular)
        if shape_coords[1][0] - shape_coords[0][0] != 0:  # Not purely vertical
            # Horizontal/diagonal edge - offset in Y
            for lane_idx in range(num_lanes):
                y_offset = 1.75 if lane_idx == 0 else -1.75
                lane_shape = " ".join([f"{x:.2f},{y+y_offset:.2f}" for x, y in shape_coords])
                xml.append(f'    <lane id="{edge_id}_{lane_idx}" index="{lane_idx}" speed="{speed}" length="999.00" shape="{lane_shape}"/>')
        else:  # Purely vertical edge
            # Vertical edge - offset in X
            for lane_idx in range(num_lanes):
                x_offset = 1.75 if lane_idx == 0 else -1.75
                lane_shape = " ".join([f"{x+x_offset:.2f},{y:.2f}" for x, y in shape_coords])
                xml.append(f'    <lane id="{edge_id}_{lane_idx}" index="{lane_idx}" speed="{speed}" length="999.00" shape="{lane_shape}"/>')
        
        xml.append('  </edge>')
    xml.append('')
    
    # Connections
    for from_edge, to_edge, from_lane, to_lane, direction, tl_id, link_idx in connections:
        xml.append(f'  <connection from="{from_edge}" to="{to_edge}" fromLane="{from_lane}" toLane="{to_lane}" dir="{direction}" tl="{tl_id}" linkIndex="{link_idx}"/>')
    xml.append('')
    
    # Traffic lights
    tl_configs = [
        ("TL0", "0", ["GGggGGggrrrr", "yyyyyyyyrrrr", "rrrrrrrrGGGG", "rrrrrrrryyyy"]),
        ("TL1", "36", ["GGggGGggrrrr", "yyyyyyyyrrrr", "rrrrrrrrGGGG", "rrrrrrrryyyy"]),
        ("TL2", "72", ["GGggrrrr", "yyyyrrrr", "rrrrGGGG", "rrrryyyy"]),
    ]
    
    for tl_id, offset, states in tl_configs:
        xml.append(f'  <tlLogic id="{tl_id}" type="static" programID="coordinated" offset="{offset}">')
        phases = [
            ("35", states[0], "EW_green"),
            ("5", states[1], "EW_yellow"),
            ("30", states[2], "NS_green"),
            ("5", states[3], "NS_yellow"),
        ]
        for duration, state, name in phases:
            xml.append(f'    <phase duration="{duration}" state="{state}" name="{name}"/>')
        xml.append('  </tlLogic>')
    
    xml.append('')
    xml.append('</net>')
    
    return "\n".join(xml)

if __name__ == "__main__":
    import os
    network_xml = generate_network()
    
    # Ensure directory exists
    os.makedirs("sumo", exist_ok=True)
    
    # Write file
    with open("sumo/network.net.xml", "w", encoding="utf-8") as f:
        f.write(network_xml)
    
    # Verify
    with open("sumo/network.net.xml", "r", encoding="utf-8") as f:
        content = f.read()
    
    line_count = content.count('\n') + 1
    print(f"Network file generated: sumo/network.net.xml ({line_count} lines, {len(content)} bytes)")
