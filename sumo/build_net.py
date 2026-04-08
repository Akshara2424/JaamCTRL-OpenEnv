"""
build_net.py  -  Jaam Ctrl
Generates sumo/network.net.xml purely in Python.
No netconvert required.

Run from the project root:
    python build_net.py

This overwrites sumo/network.net.xml with a valid SUMO network file.
"""

import os
import math

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo", "network.net.xml")

# ---------------------------------------------------------------------------
# Junction positions
# ---------------------------------------------------------------------------
JUNCTIONS = {
    "W0": (-200,    0,  "dead_end"),
    "E2": (1200,    0,  "dead_end"),
    "N0": (0,     200,  "dead_end"),
    "S0": (0,    -200,  "dead_end"),
    "N1": (500,   200,  "dead_end"),
    "S1": (500,  -200,  "dead_end"),
    "N2": (1000,  200,  "dead_end"),
    "S2": (1000, -200,  "dead_end"),
    "J0": (0,       0,  "traffic_light"),
    "J1": (500,     0,  "traffic_light"),
    "J2": (1000,    0,  "traffic_light"),
}

# ---------------------------------------------------------------------------
# Edges: (from, to, numLanes, speed, priority)
# ---------------------------------------------------------------------------
EDGES = [
    ("W0J0", "W0", "J0", 2, 13.89, 9),
    ("J0J1", "J0", "J1", 2, 13.89, 9),
    ("J1J0", "J1", "J0", 2, 13.89, 9),
    ("J1J2", "J1", "J2", 2, 13.89, 9),
    ("J2J1", "J2", "J1", 2, 13.89, 9),
    ("J2E2", "J2", "E2", 2, 13.89, 9),
    ("N0J0", "N0", "J0", 2, 11.11, 7),
    ("J0N0", "J0", "N0", 2, 11.11, 7),
    ("S0J0", "S0", "J0", 2, 11.11, 7),
    ("J0S0", "J0", "S0", 2, 11.11, 7),
    ("N1J1", "N1", "J1", 2, 11.11, 7),
    ("J1N1", "J1", "N1", 2, 11.11, 7),
    ("S1J1", "S1", "J1", 2, 11.11, 7),
    ("J1S1", "J1", "S1", 2, 11.11, 7),
    ("N2J2", "N2", "J2", 2, 11.11, 7),
    ("J2N2", "J2", "N2", 2, 11.11, 7),
    ("S2J2", "S2", "J2", 2, 11.11, 7),
    ("J2S2", "J2", "S2", 2, 11.11, 7),
]

# ---------------------------------------------------------------------------
# Connections: (fromEdge, toEdge, fromLane, toLane, direction, state)
# direction: s=straight, r=right, l=left, t=turn
# state: M=major, m=minor (yield)
# ---------------------------------------------------------------------------
CONNECTIONS = [
    # J0 from West (W0J0)
    ("W0J0", "J0J1", 0, 0, "s", "M"),
    ("W0J0", "J0J1", 1, 1, "s", "M"),
    ("W0J0", "J0S0", 0, 0, "r", "m"),
    ("W0J0", "J0N0", 1, 0, "l", "m"),
    # J0 from East (J1J0)
    ("J1J0", "J0N0", 0, 0, "r", "m"),
    ("J1J0", "J0S0", 1, 0, "l", "m"),
    # J0 from North (N0J0)
    ("N0J0", "J0J1", 0, 0, "r", "m"),
    ("N0J0", "J0S0", 1, 0, "s", "M"),
    # J0 from South (S0J0)
    ("S0J0", "J0J1", 1, 0, "l", "m"),
    ("S0J0", "J0N0", 0, 0, "s", "M"),
    # J1 from West (J0J1)
    ("J0J1", "J1J2", 0, 0, "s", "M"),
    ("J0J1", "J1J2", 1, 1, "s", "M"),
    ("J0J1", "J1S1", 0, 0, "r", "m"),
    ("J0J1", "J1N1", 1, 0, "l", "m"),
    # J1 from East (J2J1)
    ("J2J1", "J1J0", 0, 0, "s", "M"),
    ("J2J1", "J1J0", 1, 1, "s", "M"),
    ("J2J1", "J1N1", 0, 0, "r", "m"),
    ("J2J1", "J1S1", 1, 0, "l", "m"),
    # J1 from North (N1J1)
    ("N1J1", "J1J2", 0, 0, "r", "m"),
    ("N1J1", "J1J0", 1, 0, "l", "m"),
    # J1 from South (S1J1)
    ("S1J1", "J1J2", 1, 0, "l", "m"),
    ("S1J1", "J1J0", 0, 0, "r", "m"),
    # J2 from West (J1J2)
    ("J1J2", "J2E2", 0, 0, "s", "M"),
    ("J1J2", "J2E2", 1, 1, "s", "M"),
    ("J1J2", "J2S2", 0, 0, "r", "m"),
    ("J1J2", "J2N2", 1, 0, "l", "m"),
    # J2 from North (N2J2)
    ("N2J2", "J2E2", 0, 0, "r", "m"),
    ("N2J2", "J2J1", 1, 0, "l", "m"),
    # J2 from South (S2J2)
    ("S2J2", "J2E2", 1, 0, "l", "m"),
    ("S2J2", "J2J1", 0, 0, "r", "m"),
]

# ---------------------------------------------------------------------------
# TLS programs: (junction_id, offset)
# Phase order: EW-green 35s, EW-yellow 5s, NS-green 30s, NS-yellow 5s
# ---------------------------------------------------------------------------
TLS = [
    ("J0", 0),
    ("J1", 36),
    ("J2", 72),
]

LANE_WIDTH = 3.2


def edge_shape(eid, fx, fy, tx, ty, num_lanes):
    """Return the shape string for an edge (straight line, offset for lanes)."""
    dx = tx - fx
    dy = ty - fy
    length = math.hypot(dx, dy)
    if length == 0:
        return f"{fx:.2f},{fy:.2f} {tx:.2f},{ty:.2f}"
    # Perpendicular offset: half total road width
    offset = (num_lanes * LANE_WIDTH) / 2.0
    nx = -dy / length * offset
    ny =  dx / length * offset
    return f"{fx+nx:.2f},{fy+ny:.2f} {tx+nx:.2f},{ty+ny:.2f}"


def lane_shape(fx, fy, tx, ty, lane_idx, num_lanes):
    dx = tx - fx
    dy = ty - fy
    length = math.hypot(dx, dy)
    if length == 0:
        return f"{fx:.2f},{fy:.2f} {tx:.2f},{ty:.2f}"
    # Lane 0 = right/kerb, lane N-1 = left/centre
    # Offset from road centre: centre of each lane
    road_half = (num_lanes * LANE_WIDTH) / 2.0
    lane_centre = road_half - (lane_idx + 0.5) * LANE_WIDTH
    nx = -dy / length * lane_centre
    ny =  dx / length * lane_centre
    x1, y1 = fx + nx, fy + ny
    x2, y2 = tx + nx, ty + ny
    return f"{x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f}"


def junction_shape(x, y, size=8.0):
    """Square approximation for junction shape."""
    s = size
    return (f"{x-s:.2f},{y+s:.2f} {x+s:.2f},{y+s:.2f} "
            f"{x+s:.2f},{y-s:.2f} {x-s:.2f},{y-s:.2f}")


def build_tls_state(junction_id, phase_name, all_conns):
    """
    Build TLS state string for a junction.
    EW connections (from/to arterial) get G/y, NS connections get r/G/y.
    Length = number of connections at this junction.
    """
    # Gather connections at this junction (connections whose from-edge ends at junction)
    junc_conns = []
    for (fe, te, fl, tl, d, s) in all_conns:
        # Determine which junction this connection belongs to by checking
        # which signal junction the from-edge leads to
        pass

    # Simple approach: determine from edge topology
    # EW edges at each junction
    ew_edges = {
        "J0": {"W0J0", "J1J0"},
        "J1": {"J0J1", "J2J1"},
        "J2": {"J1J2"},
    }
    ns_edges = {
        "J0": {"N0J0", "S0J0"},
        "J1": {"N1J1", "S1J1"},
        "J2": {"N2J2", "S2J2"},
    }

    from_edges_ew = ew_edges.get(junction_id, set())
    from_edges_ns = ns_edges.get(junction_id, set())

    # Connections at this junction
    my_conns = [c for c in all_conns if _conn_junction(c[0]) == junction_id]

    if phase_name == "EW_green":
        state = ""
        for c in my_conns:
            fe = c[0]
            if fe in from_edges_ew:
                state += "G" if c[4] == "s" else "g"
            else:
                state += "r"
    elif phase_name == "EW_yellow":
        state = ""
        for c in my_conns:
            fe = c[0]
            state += "y" if fe in from_edges_ew else "r"
    elif phase_name == "NS_green":
        state = ""
        for c in my_conns:
            fe = c[0]
            if fe in from_edges_ns:
                state += "G" if c[4] == "s" else "g"
            else:
                state += "r"
    elif phase_name == "NS_yellow":
        state = ""
        for c in my_conns:
            fe = c[0]
            state += "y" if fe in from_edges_ns else "r"
    else:
        state = "r" * len(my_conns)

    return state if state else "rrrrrrrrrr"[:len(my_conns)] or "rrrr"


def _conn_junction(from_edge):
    """Which traffic-light junction does this from_edge lead to?"""
    mapping = {
        "W0J0": "J0", "J1J0": "J0", "N0J0": "J0", "S0J0": "J0",
        "J0J1": "J1", "J2J1": "J1", "N1J1": "J1", "S1J1": "J1",
        "J1J2": "J2", "N2J2": "J2", "S2J2": "J2",
    }
    return mapping.get(from_edge, None)


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------
def write_network():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # Build edge lookup
    edge_info = {e[0]: e for e in EDGES}  # id -> tuple

    # Build per-junction connection index and link indices
    junc_conns = {}   # junc_id -> list of conn tuples
    for c in CONNECTIONS:
        jid = _conn_junction(c[0])
        if jid:
            junc_conns.setdefault(jid, []).append(c)

    # Assign global link indices per junction
    junc_link_index = {}  # junc_id -> {(from,to,fl,tl): idx}
    for jid, conns in junc_conns.items():
        junc_link_index[jid] = {}
        for i, c in enumerate(conns):
            junc_link_index[jid][(c[0], c[1], c[2], c[3])] = i

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<net version="1.16" junctionCornerDetail="5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">')
    lines.append('')
    lines.append('    <location netOffset="0.00,0.00" convBoundary="-200.00,-200.00,1200.00,200.00" origBoundary="-200.00,-200.00,1200.00,200.00" projParameter="!"/>')
    lines.append('')

    # --- Edge types ---
    lines.append('    <type id="arterial" priority="9" numLanes="2" speed="13.89" allow="all"/>')
    lines.append('    <type id="crossroad" priority="7" numLanes="2" speed="11.11" allow="all"/>')
    lines.append('')

    # --- Edges + lanes ---
    for (eid, frm, to, num_lanes, speed, pri) in EDGES:
        fx, fy, _ = JUNCTIONS[frm]
        tx, ty, _ = JUNCTIONS[to]
        length = max(1.0, math.hypot(tx - fx, ty - fy))
        eshape = edge_shape(eid, fx, fy, tx, ty, num_lanes)
        lines.append(f'    <edge id="{eid}" from="{frm}" to="{to}" priority="{pri}" numLanes="{num_lanes}" speed="{speed:.2f}" shape="{eshape}" spreadType="center">')
        for li in range(num_lanes):
            lshape = lane_shape(fx, fy, tx, ty, li, num_lanes)
            lines.append(f'        <lane id="{eid}_{li}" index="{li}" speed="{speed:.2f}" length="{length:.2f}" shape="{lshape}"/>')
        lines.append('    </edge>')
    lines.append('')

    # --- TLS logic ---
    for (jid, offset) in TLS:
        conns_here = junc_conns.get(jid, [])
        n = len(conns_here)

        def state(phase):
            ew = {"W0J0","J1J0","J0J1","J2J1","J1J2"}
            ns = {"N0J0","S0J0","N1J1","S1J1","N2J2","S2J2"}
            s = ""
            for c in conns_here:
                fe = c[0]
                d  = c[4]
                if phase == "EW_green":
                    s += ("G" if d == "s" else "g") if fe in ew else "r"
                elif phase == "EW_yellow":
                    s += "y" if fe in ew else "r"
                elif phase == "NS_green":
                    s += ("G" if d == "s" else "g") if fe in ns else "r"
                elif phase == "NS_yellow":
                    s += "y" if fe in ns else "r"
            return s or "G" * n

        lines.append(f'    <tlLogic id="{jid}" type="static" programID="coordinated" offset="{offset}">')
        lines.append(f'        <phase duration="35" state="{state("EW_green")}"  name="EW_green"/>')
        lines.append(f'        <phase duration="5"  state="{state("EW_yellow")}" name="EW_yellow"/>')
        lines.append(f'        <phase duration="30" state="{state("NS_green")}"  name="NS_green"/>')
        lines.append(f'        <phase duration="5"  state="{state("NS_yellow")}" name="NS_yellow"/>')
        lines.append(f'    </tlLogic>')
    lines.append('')

    # --- Junctions ---
    for jid, (x, y, jtype) in JUNCTIONS.items():
        shape = junction_shape(x, y)
        if jtype == "dead_end":
            # Find incoming lanes (edges that end at this node)
            inc = [f"{e[0]}_{li}" for e in EDGES if e[2] == jid for li in range(e[3])]
            inc_str = " ".join(inc)
            lines.append(f'    <junction id="{jid}" type="dead_end" x="{x:.2f}" y="{y:.2f}" incLanes="{inc_str}" intLanes="" shape="{shape}"/>')
        else:
            inc = [f"{e[0]}_{li}" for e in EDGES if e[2] == jid for li in range(e[3])]
            inc_str = " ".join(inc)
            # Internal lanes (simplified naming)
            conns_here = junc_conns.get(jid, [])
            int_lanes = " ".join(f":{jid}_{i}_0" for i in range(len(conns_here)))
            lines.append(f'    <junction id="{jid}" type="traffic_light" x="{x:.2f}" y="{y:.2f}" tl="{jid}" incLanes="{inc_str}" intLanes="{int_lanes}" shape="{shape}"/>')
    lines.append('')

    # --- Connections ---
    for (fe, te, fl, tl, d, s) in CONNECTIONS:
        jid = _conn_junction(fe)
        if jid is None:
            continue
        idx = junc_link_index[jid].get((fe, te, fl, tl), 0)
        lines.append(f'    <connection from="{fe}" to="{te}" fromLane="{fl}" toLane="{tl}" via=":{jid}_{idx}_0" tl="{jid}" linkIndex="{idx}" dir="{d}" state="{s}"/>')
    lines.append('')
    lines.append('</net>')

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Written: {OUT}")
    print(f"  {len(EDGES)} edges, {len(CONNECTIONS)} connections, {len(TLS)} TLS junctions")


if __name__ == "__main__":
    write_network()
    print("Done. Run: sumo -c sumo/config.sumocfg")
