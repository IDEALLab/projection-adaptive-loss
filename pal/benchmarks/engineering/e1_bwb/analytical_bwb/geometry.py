"""Analytical BWB geometry from bicubic Bezier patches (upper/lower, right half-span).

Each airfoil section is one cubic Bezier, all stations lie in the z=0 chord plane.
"""

import math

import numpy as np

# Inner airfoil fitted to VTK mesh slices (RMSE 2.7 mm/m), symmetric (upper = -lower).
AIRFOIL_CP_INNER = np.array([
    [0.000000, 0.000000],   # P0, LE
    [0.142853, 0.204706],   # P1
    [0.915611, 0.010000],   # P2, raised to avoid TE tangency
    [1.000000, 0.000000],   # P3, TE
])

# Tip airfoil fitted to CST (RMSE 2.0 mm/m, t/c = 11.68%).
AIRFOIL_CP_TIP = np.array([
    [0.000000, 0.000000],   # P0, LE
    [0.075877, 0.141479],   # P1
    [0.950000, 0.010000],   # P2, raised to avoid TE tangency
    [1.000000, 0.000000],   # P3, TE
])

# Per-station CP lookup (stations 0-2 = inner, station 3 = tip).
_CPS = [AIRFOIL_CP_INNER] * 3 + [AIRFOIL_CP_TIP]


def compute_stations(B1, B2, B3, C1=1000, C2=720, C3=280, C4=90,
                     S1=60, S2=50, S3=40, L=1.0):
    """Compute the 4 spanwise station positions (right half-span).

    All length inputs are in mm (at unit scale C1=1000 mm).
    L is the isotropic scale in meters (L=1 -> C1=1 m).

    Returns dict with arrays of length 4:
        y, spanwise position  [m]
        x_le, leading-edge x     [m]
        x_te, trailing-edge x    [m]
        C, local chord        [m]
        z, chord-plane z offset (always 0, no dihedral)  [m]
    """
    scale = L / C1  # mm -> physical meters

    B = np.array([B1, B2, B3]) * scale
    C = np.array([C1, C2, C3, C4]) * scale
    S_rad = np.radians([S1, S2, S3])

    y = np.zeros(4)
    x_le = np.zeros(4)
    for i in range(3):
        y[i + 1] = y[i] + B[i]
        x_le[i + 1] = x_le[i] + B[i] * math.tan(S_rad[i])

    x_te = x_le + C

    z = np.zeros(4)

    return {"y": y, "x_le": x_le, "x_te": x_te, "C": C, "B": B, "z": z}


def _tangent_sweeps(S1, S2, S3):
    """Compute In/Out LE and TE sweep angles + strengths at each station."""
    S12 = (S1 + S2) / 2.0
    S23 = (S2 + S3) / 2.0

    return {
        0: {"out_le": 0.0,  "out_te": 0.0,
            "out_le_str": 0.95, "out_te_str": 0.50},
        1: {"in_le": S12,   "in_te": -S12,
            "out_le": S12,  "out_te": -S12,
            "in_le_str": 0.40, "in_te_str": 0.30,
            "out_le_str": 0.40, "out_te_str": 0.30},
        2: {"in_le": S23,   "in_te": -S12 * 0.6,
            "out_le": S23,  "out_te": -S12 * 0.6,
            "in_le_str": 0.95, "in_te_str": 0.20,
            "out_le_str": 1.00, "out_te_str": 0.20},
        3: {"in_le": S3,    "in_te": 0.7 * S3,
            "in_le_str": 1.00, "in_te_str": 1.00},
    }


def _airfoil_3d(x_le, y, chord, z_base, cp, sign=1):
    """4 control points of one Bezier airfoil at a station.

    sign=+1 : upper surface  (z = z_base + fz*chord)
    sign=-1 : lower surface  (z = z_base - fz*chord)

    Returns (4, 3) array with columns [x, y, z].
    """
    pts = np.zeros((4, 3))
    for j in range(4):
        fx, fz = cp[j]
        pts[j] = [x_le + fx * chord, y, z_base + sign * fz * chord]
    return pts


def _tangent_vec(alpha_le_deg, alpha_te_deg, frac_x, frac_z,
                 str_le=1.0, str_te=1.0):
    """Spanwise tangent direction for one chordwise CP.

    frac_z is the signed z/chord of the airfoil at this CP (negative for
    lower-surface CPs).
    """
    tan_le = math.tan(math.radians(alpha_le_deg)) * str_le
    tan_te = math.tan(math.radians(alpha_te_deg)) * str_te
    tx = tan_le + frac_x * (tan_te - tan_le)
    ty = str_le + frac_x * (str_te - str_le)
    tz = frac_z * (tan_te - tan_le)
    return np.array([tx, ty, tz])


def _build_wing_patch(stations, tangents, k, cp_k, cp_k1, sign=1):
    """Build a 4x4x3 control net for wing panel k->k+1.

    cp_k, cp_k1 : (4,2) airfoil CP arrays for station k and k+1.
    sign        : +1 upper surface, -1 lower surface.
    """
    dy = float(stations["B"][k])

    row0 = _airfoil_3d(stations["x_le"][k],   stations["y"][k],   stations["C"][k],   0.0, cp_k,  sign)
    row3 = _airfoil_3d(stations["x_le"][k+1], stations["y"][k+1], stations["C"][k+1], 0.0, cp_k1, sign)

    tk  = tangents[k]
    tk1 = tangents[k + 1]

    row1 = np.zeros((4, 3))
    row2 = np.zeros((4, 3))
    for j in range(4):
        fx, fz = cp_k[j]
        fz_signed = sign * fz
        t_out = _tangent_vec(tk["out_le"], tk["out_te"], fx, fz_signed,
                             tk["out_le_str"], tk["out_te_str"])

        fx1, fz1 = cp_k1[j]
        fz1_signed = sign * fz1
        t_in = _tangent_vec(tk1["in_le"], tk1["in_te"], fx1, fz1_signed,
                            tk1["in_le_str"], tk1["in_te_str"])

        row1[j] = row0[j] + (dy / 3.0) * t_out
        row2[j] = row3[j] - (dy / 3.0) * t_in

    row1[3] = row0[3] + (1.0 / 3.0) * (row3[3] - row0[3])
    row2[3] = row0[3] + (2.0 / 3.0) * (row3[3] - row0[3])

    return np.stack([row0, row1, row2, row3], axis=0)  # (4, 4, 3)


def compute_control_nets(B1=150, B2=100, B3=500, C1=1000, C2=720, C3=280,
                         C4=90, S1=60, S2=50, S3=40, L=1.0):
    """Compute all control nets for the BWB upper/lower right quarter.

    Returns dict mapping patch name -> (4, 4, 3) numpy array.
    Names: patch_{k}_{upper|lower}  for k in 0,1,2
    """
    stations = compute_stations(B1, B2, B3, C1, C2, C3, C4, S1, S2, S3, L)
    tangents = _tangent_sweeps(S1, S2, S3)

    nets = {}
    for k in range(3):
        cp_k  = _CPS[k]
        cp_k1 = _CPS[k + 1]

        for sign, surf in [(1, "upper"), (-1, "lower")]:
            name = f"patch_{k}_{surf}"
            nets[name] = _build_wing_patch(
                stations, tangents, k, cp_k, cp_k1, sign)

    return nets


def build_yaml_config(control_nets, stations, resolution=128):
    """Build a complete geometry YAML config dict for the BWB.

    All stations at z=0.  Mirror xz (y->-y) for full aircraft.
    """
    all_pts = np.concatenate([net.reshape(-1, 3) for net in control_nets.values()])
    x_max = float(all_pts[:, 0].max()) * 1.3 + 0.1
    y_max = float(all_pts[:, 1].max()) * 1.3 + 0.1
    z_max = float(max(abs(all_pts[:, 2].min()), all_pts[:, 2].max())) * 2.5 + 0.05

    def _net_to_list(net):
        return [[[float(v) for v in pt] for pt in row] for row in net]

    shapes = {}
    K_UNION = 50.0
    K_INTERSECT = 50.0

    y = [float(v) for v in stations["y"]]
    x_big = float(x_max) * 2
    z_big = float(z_max) * 2

    upper_right_parts = []
    lower_right_parts = []

    x_te_vals = [float(v) for v in stations["x_te"]]

    for k_idx in range(3):
        margin = 0.05
        y_lo = y[k_idx]     - (margin if k_idx == 0 else 0)
        y_hi = y[k_idx + 1] + (margin if k_idx == 2 else 0)
        y_center = (y_lo + y_hi) / 2.0
        y_size   = y_hi - y_lo

        # y-slab: clips spanwise extent only.
        slab_name = f"slab_{k_idx}"
        shapes[slab_name] = {
            "type": "box",
            "size": [x_big * 2, y_size, z_big * 2],
            "center": [x_big / 2, y_center, 0],
        }

        # TE half-space x < x_te(y): box rotated by atan2(-dx_te, dy) through the TE midpoint.
        x_te_k  = x_te_vals[k_idx]
        x_te_k1 = x_te_vals[k_idx + 1]
        dx_te   = x_te_k1 - x_te_k
        dy_te   = y[k_idx + 1] - y[k_idx]
        alpha_deg = math.degrees(math.atan2(-dx_te, dy_te))
        x_te_mid = round((x_te_k + x_te_k1) / 2.0, 5)
        y_te_mid = round((y[k_idx] + y[k_idx + 1]) / 2.0, 5)

        large = x_big * 10
        te_raw_name  = f"te_raw_{k_idx}"
        te_rot_name  = f"te_rot_{k_idx}"
        te_clip_name = f"te_clip_{k_idx}"
        shapes[te_raw_name] = {
            "type": "box",
            "size": [large * 2, large * 2, z_big * 4],
            "center": [-large, 0, 0],
        }
        shapes[te_rot_name] = {
            "type": "rotate",
            "shape": te_raw_name,
            "axis": "z",
            "angle": round(alpha_deg, 4),
        }
        shapes[te_clip_name] = {
            "type": "translate",
            "shape": te_rot_name,
            "offset": [x_te_mid, y_te_mid, 0.0],
        }

        for surf, parts_list in [("upper", upper_right_parts),
                                  ("lower", lower_right_parts)]:
            flip = (surf == "upper")
            pname = f"patch_{k_idx}_{surf}"
            shapes[pname] = {
                "type": "bezier_surface",
                "control_net": _net_to_list(control_nets[pname]),
                "flip": flip,
            }
            # Step 1: spanwise y-slab (smooth intersection preserves SDF quality)
            y_clip_name = f"panel_{k_idx}_{surf}_y"
            shapes[y_clip_name] = {
                "type": "smooth_intersection",
                "shapes": [pname, slab_name],
                "k": K_INTERSECT,
            }
            # Step 2: chordwise TE clip (sharp, TE is a genuine geometry boundary)
            clipped_name = f"panel_{k_idx}_{surf}"
            shapes[clipped_name] = {
                "type": "intersection",
                "shapes": [y_clip_name, te_clip_name],
            }
            parts_list.append(clipped_name)

    shapes["upper_right"] = {
        "type": "smooth_union",
        "shapes": upper_right_parts,
        "k": K_UNION,
    }
    shapes["lower_right"] = {
        "type": "smooth_union",
        "shapes": lower_right_parts,
        "k": K_UNION,
    }
    shapes["right_half"] = {
        "type": "intersection",
        "shapes": ["upper_right", "lower_right"],
    }
    shapes["left_half"] = {
        "type": "mirror",
        "shape": "right_half",
        "plane": "xz",
        "offset": 0,
    }
    shapes["bwb"] = {
        "type": "smooth_union",
        "shapes": ["right_half", "left_half"],
        "k": 150.0,
    }

    return {
        "name": "bwb_analytical",
        "bounds": {
            "x": [round(-0.05 * x_max, 4), round(x_max, 4)],
            "y": [round(-y_max, 4), round(y_max, 4)],
            "z": [round(-z_max, 4), round(z_max, 4)],
            "resolution": resolution,
            "batch_size": 1,
        },
        "shapes": shapes,
        "output": "bwb",
    }
