#!/usr/bin/env python3
"""
Sort SUMO routes.rou.xml by departure time.
Interleaves flows and vehicles in chronological order.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

def sort_routes_file(filepath):
    """Sort routes file by departure time, interleaving flows and vehicles."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Collect flows and vehicles
    flows = []
    vehicles = []
    other_elements = []
    
    for element in root:
        if element.tag == 'flow':
            begin_time = int(element.get('begin', '0'))
            flows.append((begin_time, element))
        elif element.tag == 'vehicle':
            depart_time = int(element.get('depart', '0'))
            vehicles.append((depart_time, element))
        else:
            other_elements.append(element)
    
    # Sort by departure time
    flows.sort(key=lambda x: x[0])
    vehicles.sort(key=lambda x: x[0])
    
    # Merge flows and vehicles in chronological order
    sorted_departures = []
    f_idx, v_idx = 0, 0
    
    while f_idx < len(flows) or v_idx < len(vehicles):
        if f_idx >= len(flows):
            sorted_departures.append((vehicles[v_idx][0], 'vehicle', vehicles[v_idx][1]))
            v_idx += 1
        elif v_idx >= len(vehicles):
            sorted_departures.append((flows[f_idx][0], 'flow', flows[f_idx][1]))
            f_idx += 1
        else:
            if flows[f_idx][0] <= vehicles[v_idx][0]:
                sorted_departures.append((flows[f_idx][0], 'flow', flows[f_idx][1]))
                f_idx += 1
            else:
                sorted_departures.append((vehicles[v_idx][0], 'vehicle', vehicles[v_idx][1]))
                v_idx += 1
    
    # Rebuild root
    root.clear()
    
    # Re-add other elements (vTypes, routes, etc.) in original order
    for elem in other_elements:
        root.append(elem)
    
    # Add sorted flows and vehicles interleaved
    for _, elem_type, elem in sorted_departures:
        root.append(elem)
    
    # Write back preserving formatting
    tree.write(filepath, encoding='UTF-8', xml_declaration=True)
    print(f"✓ Sorted {len(flows)} flows and {len(vehicles)} vehicles by departure time")
    print(f"  Flows start at: {flows[0][0] if flows else 'N/A'}")
    print(f"  Vehicles depart at: {vehicles[0][0] if vehicles else 'N/A'}")

if __name__ == '__main__':
    routes_file = Path(__file__).parent / 'sumo' / 'routes.rou.xml'
    if routes_file.exists():
        sort_routes_file(routes_file)
    else:
        print(f"File not found: {routes_file}")
