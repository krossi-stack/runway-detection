"""
Auto-label runway segmentation masks from X-Plane telemetry.

For each captured session, projects the target runway's 4 corners into each
frame using the aircraft pose and X-Plane's camera FOV, then saves YOLO
segmentation labels (normalized polygon coordinates).

Usage:
  python scripts/auto_label.py --session 20260421_143000 --airport KSEA --runway 16L
  python scripts/auto_label.py --session 20260421_143000 --runway-coords 47.464,-122.308,47.444,-122.314

The second form lets you specify threshold coordinates directly if you don't
have the apt.dat file.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABELS_DIR,
    RAW_FRAMES_DIR,
    XPLANE_APT_DAT,
    XPLANE_FOV_DEG,
    XPLANE_FOV_V_DEG,
    XPLANE_TITLE_BAR_PX,
)


# ---------------------------------------------------------------------------
# acf parsing
# ---------------------------------------------------------------------------

def parse_pilot_eye(acf_path: str) -> np.ndarray:
    """Read pilot eye position (x_right, y_up, z_fwd) in meters from a .acf file."""
    eye = [0.0, 0.0, 0.0]
    with open(acf_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if line.startswith("P acf/_pe_xyz/0"):
                eye[0] = float(line.split()[-1])
            elif line.startswith("P acf/_pe_xyz/1"):
                eye[1] = float(line.split()[-1])
            elif line.startswith("P acf/_pe_xyz/2"):
                eye[2] = float(line.split()[-1])
    # X-Plane acf: x=right, y=up, z=forward -> body frame: fwd, right, up
    return np.array([eye[2], eye[0], eye[1]])  # (fwd, right, up)


# ---------------------------------------------------------------------------
# apt.dat parsing
# ---------------------------------------------------------------------------

def parse_apt_dat(apt_dat_path: str, target_icao: str) -> list[dict]:
    """Parse runway records for a given airport from apt.dat.

    Returns a list of dicts, each with:
      id1, lat1, lon1, id2, lat2, lon2, width_m
    """
    runways = []
    in_target = False

    with open(apt_dat_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue

            code = parts[0]

            if code in ("1", "16", "17"):
                in_target = (len(parts) >= 5 and parts[4] == target_icao)
                continue

            if in_target and code == "100" and len(parts) >= 26:
                runways.append({
                    "width_m": float(parts[1]),
                    "id1": parts[8],
                    "lat1": float(parts[9]),
                    "lon1": float(parts[10]),
                    "elev1_ft": float(parts[11]),
                    "id2": parts[17],
                    "lat2": float(parts[18]),
                    "lon2": float(parts[19]),
                    "elev2_ft": float(parts[20]),
                })

    return runways


def find_runway(runways: list[dict], runway_id: str) -> dict | None:
    for rwy in runways:
        if rwy["id1"] == runway_id or rwy["id2"] == runway_id:
            return rwy
    return None


# ---------------------------------------------------------------------------
# Geo / projection math
# ---------------------------------------------------------------------------

def latlon_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Convert lat/lon/alt to local East-North-Up (meters) relative to ref."""
    R = 6378137.0
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    cos_ref = math.cos(math.radians(ref_lat))

    east = dlon * R * cos_ref
    north = dlat * R
    up = alt - ref_alt
    return np.array([east, north, up])


def runway_corners_enu(lat1, lon1, lat2, lon2, width_m, ref_lat, ref_lon, ref_alt=0.0,
                       elev1_m=0.0, elev2_m=0.0):
    """Compute 4 runway corners in ENU coordinates."""
    p1 = latlon_to_enu(lat1, lon1, elev1_m, ref_lat, ref_lon, ref_alt)
    p2 = latlon_to_enu(lat2, lon2, elev2_m, ref_lat, ref_lon, ref_alt)

    along = p2 - p1
    along_2d = np.array([along[0], along[1], 0.0])
    along_2d_norm = along_2d / np.linalg.norm(along_2d)

    perp = np.array([-along_2d_norm[1], along_2d_norm[0], 0.0])
    half_w = width_m / 2.0

    c1 = p1 + perp * half_w
    c2 = p1 - perp * half_w
    c3 = p2 - perp * half_w
    c4 = p2 + perp * half_w

    return np.array([c1, c2, c3, c4])


def rotation_matrix(pitch_deg, roll_deg, heading_deg):
    """Aircraft body-to-ENU rotation matrix.

    Convention: heading=0 is north, positive clockwise.
    Pitch positive = nose up. Roll positive = right wing down.
    Returns R such that v_enu = R @ v_body.
    """
    h = math.radians(heading_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    # Heading rotation (around Up axis, CW from North)
    Rh = np.array([
        [math.sin(h),  math.cos(h), 0],
        [math.cos(h), -math.sin(h), 0],
        [0,            0,           -1],
    ])

    # Pitch rotation (around right/y axis) -- nose up positive
    Rp = np.array([
        [ math.cos(p), 0, math.sin(p)],
        [ 0,           1, 0          ],
        [-math.sin(p), 0, math.cos(p)],
    ])

    # Roll rotation (around forward/x axis) -- right bank positive
    Rr = np.array([
        [1, 0,           0           ],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r),  math.cos(r)],
    ])

    return Rh @ Rp @ Rr


def project_points_to_image(points_enu, aircraft_enu, pitch, roll, heading,
                            fov_h_deg, img_w, img_h, fov_v_deg=None,
                            eye_offset=None, pitch_bias_deg=0.0, roll_bias_deg=0.0):
    """Project 3D ENU points into 2D image coordinates.

    eye_offset: (fwd, right, up) in meters from CG to pilot eye in body frame.
    """
    R_body_to_enu = rotation_matrix(pitch + pitch_bias_deg, roll + roll_bias_deg, heading)
    R_enu_to_body = R_body_to_enu.T

    if eye_offset is not None:
        fwd, right, up = eye_offset
        offset_body = np.array([fwd, right, -up])  # body: x=fwd, y=right, z=down
        aircraft_enu = aircraft_enu + R_body_to_enu @ offset_body

    # The rendered 3D area excludes the title bar
    rendered_h = img_h - XPLANE_TITLE_BAR_PX

    fx = (img_w / 2.0) / math.tan(math.radians(fov_h_deg / 2.0))
    if fov_v_deg is None:
        fov_v_deg = 2.0 * math.degrees(math.atan(math.tan(math.radians(fov_h_deg / 2.0)) * rendered_h / img_w))
    fy = (rendered_h / 2.0) / math.tan(math.radians(fov_v_deg / 2.0))

    cx = img_w / 2.0
    cy = XPLANE_TITLE_BAR_PX + rendered_h / 2.0

    # Transform all 4 corners to body frame first
    body_pts = []
    for pt in points_enu:
        p_enu = pt - aircraft_enu
        p_body = R_enu_to_body @ p_enu
        body_pts.append(np.array(p_body, dtype=float))

    # Clip polygon against near plane (fwd > near) using Sutherland-Hodgman
    near = 1.0  # 1 meter near plane
    clipped = _clip_polygon_near(body_pts, near)
    if not clipped:
        return None  # entirely behind camera

    projected = []
    for p_body in clipped:
        fwd, right, down = p_body[0], p_body[1], p_body[2]
        u = cx + fx * (right / fwd)
        v = cy + fy * (down / fwd)
        projected.append((u, v))

    return projected


def _clip_polygon_near(body_pts: list, near: float) -> list:
    """Clip a polygon (list of body-frame points) against the near plane fwd=near.

    Uses Sutherland-Hodgman clipping. Returns clipped polygon or empty list.
    """
    output = list(body_pts)
    if not output:
        return []

    clipped = []
    n = len(output)
    for i in range(n):
        a = output[i]
        b = output[(i + 1) % n]
        a_inside = a[0] >= near
        b_inside = b[0] >= near

        if a_inside:
            clipped.append(a)

        if a_inside != b_inside:
            # Compute intersection with fwd=near plane
            t = (near - a[0]) / (b[0] - a[0])
            intersection = a + t * (b - a)
            clipped.append(intersection)

    return clipped


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def points_to_yolo_seg(points, img_w, img_h, class_id=0):
    """Convert pixel coordinates to YOLO segmentation format.

    Format: class_id x1 y1 x2 y2 ... (all normalized 0-1)
    """
    parts = [str(class_id)]
    for u, v in points:
        parts.append(f"{u / img_w:.6f}")
        parts.append(f"{v / img_h:.6f}")
    return " ".join(parts)


def label_session(session_dir: Path, corners_latlon, width_m, fov_deg, img_w, img_h,
                  fov_v_deg=None, elev1_m=0.0, elev2_m=0.0, eye_offset=None,
                  pitch_bias_deg=0.0, roll_bias_deg=0.0):
    """Generate labels for all frames in a capture session."""
    frames_dir = session_dir / "frames"
    telemetry_path = session_dir / "telemetry.csv"

    if not telemetry_path.exists():
        print(f"No telemetry.csv in {session_dir}")
        return

    output_dir = Path(LABELS_DIR) / session_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    lat1, lon1, lat2, lon2 = corners_latlon
    ref_lat = (lat1 + lat2) / 2.0
    ref_lon = (lon1 + lon2) / 2.0

    rwy_corners = runway_corners_enu(lat1, lon1, lat2, lon2, width_m,
                                      ref_lat, ref_lon,
                                      elev1_m=elev1_m, elev2_m=elev2_m)

    labeled = 0
    skipped = 0

    with open(telemetry_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame"])
            frame_file = frames_dir / f"{frame_idx:06d}.jpg"
            if not frame_file.exists():
                continue

            lat = float(row["lat"])
            lon = float(row["lon"])
            alt_agl_ft = float(row["alt_agl_ft"])
            pitch = float(row["pitch"])
            roll = float(row["roll"])
            heading = float(row["heading"])

            alt_m = alt_agl_ft * 0.3048
            aircraft_enu = latlon_to_enu(lat, lon, alt_m, ref_lat, ref_lon, 0.0)

            projected = project_points_to_image(
                rwy_corners, aircraft_enu, pitch, roll, heading,
                fov_deg, img_w, img_h, fov_v_deg=fov_v_deg,
                eye_offset=eye_offset, pitch_bias_deg=pitch_bias_deg,
                roll_bias_deg=roll_bias_deg,
            )

            if projected is None or len(projected) < 3:
                skipped += 1
                continue

            pts = np.array(projected)

            # Check at least some of the polygon is in frame
            if np.all(pts[:, 0] < 0) or np.all(pts[:, 0] > img_w):
                skipped += 1
                continue
            if np.all(pts[:, 1] < 0) or np.all(pts[:, 1] > img_h):
                skipped += 1
                continue

            # Clamp to image bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)

            label_line = points_to_yolo_seg(pts.tolist(), img_w, img_h)
            label_path = output_dir / f"{frame_idx:06d}.txt"
            with open(label_path, "w") as lf:
                lf.write(label_line + "\n")

            labeled += 1

    print(f"Labeled: {labeled}  |  Skipped: {skipped}  |  Output: {output_dir}")
    return labeled, skipped


def preview_labels(session_dir: Path, label_dir: Path, num_frames=None):
    """Show a few frames with the projected runway overlay for visual QA."""
    frames_dir = session_dir / "frames"
    all_labels = sorted(label_dir.glob("*.txt"))
    label_files = all_labels if num_frames is None else all_labels[:num_frames]

    for lf in label_files:
        frame_file = frames_dir / lf.with_suffix(".jpg").name
        if not frame_file.exists():
            continue

        img = cv2.imread(str(frame_file))
        h, w = img.shape[:2]

        with open(lf) as f:
            parts = f.read().strip().split()
        coords = [float(x) for x in parts[1:]]
        points = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i + 1] * h)
            points.append([px, py])

        poly = np.array(points, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], (0, 255, 0))
        display = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        cv2.polylines(display, [poly], True, (0, 255, 0), 2)

        cv2.imshow(f"Preview: {lf.name}", display)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Auto-label runway segmentation masks")
    parser.add_argument("--session", required=True, help="Session folder name (e.g. 20260421_143000)")
    parser.add_argument("--airport", default=None, help="ICAO airport code (e.g. KSEA)")
    parser.add_argument("--runway", default=None, help="Runway ID (e.g. 16L)")
    parser.add_argument("--runway-coords", default=None,
                        help="Direct threshold coords: lat1,lon1,lat2,lon2")
    parser.add_argument("--width", type=float, default=45.0, help="Runway width in meters")
    parser.add_argument("--apt-dat", default=None, help="Path to apt.dat file")
    parser.add_argument("--fov", type=float, default=XPLANE_FOV_DEG, help="Horizontal FOV in degrees (default: XPLANE_FOV_DEG)")
    parser.add_argument("--fov-v", type=float, default=XPLANE_FOV_V_DEG, help="Vertical FOV in degrees (default: XPLANE_FOV_V_DEG)")
    parser.add_argument("--acf", default=None, help="Path to aircraft .acf file to read pilot eye position")
    parser.add_argument("--pitch-bias", type=float, default=0.0,
                        help="Degrees to add to pitch before projecting (positive = camera looks more upward). "
                             "Tune per aircraft to fix polygon height error on approach.")
    parser.add_argument("--roll-bias", type=float, default=0.0,
                        help="Degrees to add to roll before projecting (positive = camera tilts right). "
                             "Tune per aircraft to fix polygon left/right tilt error.")
    parser.add_argument("--img-width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--img-height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--preview", action="store_true", help="Show preview of labeled frames")
    args = parser.parse_args()

    session_dir = Path(RAW_FRAMES_DIR) / args.session
    if not session_dir.exists():
        print(f"Session not found: {session_dir}")
        sys.exit(1)

    elev1_m = 0.0
    elev2_m = 0.0

    if args.runway_coords:
        parts = [float(x) for x in args.runway_coords.split(",")]
        if len(parts) != 4:
            print("--runway-coords must be lat1,lon1,lat2,lon2")
            sys.exit(1)
        corners = tuple(parts)
    elif args.airport and args.runway:
        apt_path = args.apt_dat or XPLANE_APT_DAT
        if not apt_path:
            print("Set XPLANE_APT_DAT in .env or pass --apt-dat")
            sys.exit(1)
        runways = parse_apt_dat(apt_path, args.airport)
        if not runways:
            print(f"No runways found for {args.airport}")
            sys.exit(1)
        rwy = find_runway(runways, args.runway)
        if not rwy:
            available = [f"{r['id1']}/{r['id2']}" for r in runways]
            print(f"Runway {args.runway} not found. Available: {available}")
            sys.exit(1)
        corners = (rwy["lat1"], rwy["lon1"], rwy["lat2"], rwy["lon2"])
        args.width = rwy["width_m"]
        elev1_m = rwy["elev1_ft"] * 0.3048
        elev2_m = rwy["elev2_ft"] * 0.3048
        print(f"Found runway {rwy['id1']}/{rwy['id2']}: width={rwy['width_m']:.1f}m "
              f"elev={rwy['elev1_ft']:.0f}/{rwy['elev2_ft']:.0f}ft")
    else:
        print("Provide either --airport + --runway, or --runway-coords")
        sys.exit(1)

    print(f"Session: {args.session}")
    print(f"Runway: ({corners[0]:.6f}, {corners[1]:.6f}) -> ({corners[2]:.6f}, {corners[3]:.6f})")
    print(f"Width: {args.width:.1f}m  |  FOV: {args.fov:.1f}deg  |  Image: {args.img_width}x{args.img_height}")
    print()

    eye_offset = None
    if args.acf:
        eye_offset = parse_pilot_eye(args.acf)
        print(f"Pilot eye offset (fwd, right, up): {eye_offset[0]:.2f}m, {eye_offset[1]:.2f}m, {eye_offset[2]:.2f}m")

    labeled, skipped = label_session(
        session_dir, corners, args.width, args.fov,
        args.img_width, args.img_height, fov_v_deg=args.fov_v,
        elev1_m=elev1_m, elev2_m=elev2_m, eye_offset=eye_offset,
        pitch_bias_deg=args.pitch_bias, roll_bias_deg=args.roll_bias,
    )

    if args.preview:
        label_dir = Path(LABELS_DIR) / args.session
        preview_labels(session_dir, label_dir)


if __name__ == "__main__":
    main()
