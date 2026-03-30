import cv2
import numpy as np
from collections import deque

INPUT_VIDEO      = "video.mp4"
OUTPUT_VIDEO     = "output.mp4"
NUM_POINTS       = 10
WIN_HALF         = 15
PYRAMID_LEVELS   = 4
MAX_ITER         = 30
CONVERGENCE_EPS  = 0.01
TRAIL_LENGTH     = 30
REDETECT_EVERY   = 20
MIN_ALIVE        = 4
LINE_COLOR       = (0, 0, 255)
POINT_COLOR      = (0, 0, 255)
LINE_THICKNESS   = 2
DOT_RADIUS       = 4
INPUT_SCALE      = 0.33  

def image_gradients(img: np.ndarray):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / 8.0
    Ky = Kx.T
    Ix = cv2.filter2D(img, cv2.CV_32F, Kx)
    Iy = cv2.filter2D(img, cv2.CV_32F, Ky)
    return Ix, Iy

def gaussian_pyramid(img: np.ndarray, levels: int):
    pyr = [img.astype(np.float32)]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        pyr.append(img.astype(np.float32))
    return pyr

def lk_solve(Ix_p, Iy_p, It_p, eps=1e-6):
    Ixx = float((Ix_p * Ix_p).sum())
    Ixy = float((Ix_p * Iy_p).sum())
    Iyy = float((Iy_p * Iy_p).sum())
    Ixt = float((Ix_p * It_p).sum())
    Iyt = float((Iy_p * It_p).sum())

    det = Ixx * Iyy - Ixy * Ixy
    if abs(det) < eps:
        return 0.0, 0.0, False

    u = (Iyy * (-Ixt) - Ixy * (-Iyt)) / det
    v = (Ixx * (-Iyt) - Ixy * (-Ixt)) / det
    return u, v, True

def track_points(prev_gray: np.ndarray,
                 curr_gray: np.ndarray,
                 points: np.ndarray,
                 win_half: int = WIN_HALF,
                 levels: int   = PYRAMID_LEVELS,
                 max_iter: int = MAX_ITER,
                 eps: float    = CONVERGENCE_EPS):
    N = len(points)
    if N == 0:
        return points.copy(), np.zeros(0, dtype=np.uint8)

    pyr_prev = gaussian_pyramid(prev_gray, levels)
    pyr_curr = gaussian_pyramid(curr_gray, levels)

    H_full, W_full = prev_gray.shape
    disp = np.zeros((N, 2), dtype=np.float32)

    for lv in range(levels - 1, -1, -1):
        scale  = 2.0 ** lv
        Ip     = pyr_prev[lv]
        Ic     = pyr_curr[lv]
        h, w   = Ip.shape
        wh     = max(win_half - lv, 3)

        Ix, Iy = image_gradients(Ip)
        pts_lv  = points / scale
        disp_lv = disp   / scale

        for i in range(N):
            x0, y0 = pts_lv[i]
            dx, dy  = float(disp_lv[i, 0]), float(disp_lv[i, 1])

            for _ in range(max_iter):
                px0 = int(round(x0));  py0 = int(round(y0))
                cx  = int(round(x0 + dx));  cy = int(round(y0 + dy))

                if (py0 - wh < 0 or py0 + wh + 1 > h or
                        px0 - wh < 0 or px0 + wh + 1 > w or
                        cy  - wh < 0 or cy  + wh + 1 > h or
                        cx  - wh < 0 or cx  + wh + 1 > w):
                    break

                sl_p = np.s_[py0 - wh: py0 + wh + 1, px0 - wh: px0 + wh + 1]
                sl_c = np.s_[cy  - wh: cy  + wh + 1, cx  - wh: cx  + wh + 1]

                It_p = Ic[sl_c].astype(np.float32) - Ip[sl_p].astype(np.float32)
                u, v, ok = lk_solve(Ix[sl_p], Iy[sl_p], It_p)

                dx += u;  dy += v
                if (u*u + v*v) < eps * eps:
                    break

            disp_lv[i] = [dx, dy]

        disp = disp_lv * scale

    new_pts = points + disp

    status = np.ones(N, dtype=np.uint8)
    for i in range(N):
        nx, ny = new_pts[i]
        if not (0 <= nx < W_full and 0 <= ny < H_full):
            status[i] = 0
        elif np.linalg.norm(disp[i]) > min(H_full, W_full) * 0.12:
            status[i] = 0

    return new_pts.astype(np.float32), status

def seed_points(gray: np.ndarray, n: int = NUM_POINTS):
    h, w = gray.shape
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[int(h * 0.35):, :] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=n,
        qualityLevel=0.05,
        minDistance=max(gray.shape) // (n + 1),
        blockSize=7,
        mask=mask,
    )
    if pts is None or len(pts) == 0:
        xs = np.linspace(w * 0.1, w * 0.9, n, dtype=np.float32)
        ys = np.full(n, h * 0.6, dtype=np.float32)
        pts = np.stack([xs, ys], axis=1)
    else:
        pts = pts.reshape(-1, 2)

    if len(pts) >= n:
        pts = pts[:n]
    return pts.astype(np.float32)

def draw_trails(frame: np.ndarray,
                trails: list,
                curr_pts: np.ndarray,
                status: np.ndarray):
    for i, (trail, alive) in enumerate(zip(trails, status)):
        if alive and i < len(curr_pts):
            x, y = int(curr_pts[i, 0]), int(curr_pts[i, 1])
            trail.append((x, y))
        pts_list = list(trail)
        for j in range(1, len(pts_list)):
            cv2.line(frame, pts_list[j - 1], pts_list[j],
                     LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)
        if trail:
            cv2.circle(frame, trail[-1], DOT_RADIUS,
                       POINT_COLOR, -1, cv2.LINE_AA)
            
def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    frame         = cv2.resize(frame, (0, 0), fx=INPUT_SCALE, fy=INPUT_SCALE)
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr_pts  = seed_points(prev_gray, NUM_POINTS)
    trails    = [deque(maxlen=TRAIL_LENGTH) for _ in range(NUM_POINTS)]
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame     = cv2.resize(frame, (0, 0), fx=INPUT_SCALE, fy=INPUT_SCALE)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        new_pts, status = track_points(prev_gray, curr_gray, curr_pts)

        draw_trails(frame, trails, new_pts, status)
        out.write(frame)

        n_alive = int(status.sum())

        if frame_idx % REDETECT_EVERY == 0 or n_alive < MIN_ALIVE:
            new_seeds  = seed_points(curr_gray, NUM_POINTS)
            refreshed  = []
            new_trails = []

            alive_idx = [i for i, s in enumerate(status) if s == 1]
            for k, i in enumerate(alive_idx[:NUM_POINTS]):
                refreshed.append(new_pts[i])
                new_trails.append(trails[i])

            used   = len(refreshed)
            seed_q = deque(new_seeds.tolist())
            while used < NUM_POINTS and seed_q:
                candidate = np.array(seed_q.popleft(), dtype=np.float32)
                if all(np.linalg.norm(candidate - np.array(r)) > 10
                       for r in refreshed):
                    refreshed.append(candidate)
                    new_trails.append(deque(maxlen=TRAIL_LENGTH))
                    used += 1

            while len(refreshed) < NUM_POINTS:
                refreshed.append(new_seeds[0] if len(new_seeds) else np.array([0., 0.]))
                new_trails.append(deque(maxlen=TRAIL_LENGTH))

            curr_pts = np.array(refreshed[:NUM_POINTS], dtype=np.float32)
            trails   = new_trails[:NUM_POINTS]
        else:
            curr_pts = new_pts
            for i, s in enumerate(status):
                if not s:
                    trails[i].clear()

        prev_gray = curr_gray
        frame_idx += 1
    cap.release()
    out.release()
    print(f"\nOutput saved to '{OUTPUT_VIDEO}'.")


if __name__ == "__main__":
    main()