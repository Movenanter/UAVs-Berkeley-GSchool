"""Steps 4-5: custom stitching pipeline.

video -> every Nth frame -> keypoints -> matches (ratio test)
      -> transform between consecutive frames (RANSAC)
      -> chain transforms so every frame is placed relative to frame 0
      -> canvas size -> warp + overlay -> stitched image + stats
"""

import argparse
import os
import time

import cv2
import numpy as np

from frame_extraction import sample_frames

MIN_INLIERS = 20          # fewer RANSAC inliers than this = the pair failed
MAX_CANVAS_SIDE = 20000   # refuse absurd canvases caused by a bad transform
RANSAC_THRESHOLD = 3.0    # max reprojection error (px) for a match to be an inlier


# ---------------------------------------------------------------- features --

def create_detector(method, n_features):
    """ORB: fast, binary descriptors. SIFT: slower, float descriptors, more robust."""
    if method == "orb":
        return cv2.ORB_create(nfeatures=n_features)
    return cv2.SIFT_create(nfeatures=n_features)


def detect_features(detector, image):
    """Return (keypoints, descriptors) for one BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(gray, None)


def match_features(des_query, des_train, method, ratio):
    """Match descriptors with brute force k-NN and keep only distinctive matches.

    Lowe's ratio test: keep a match only if the best candidate is clearly better
    than the second best (distance < ratio * second distance). Repetitive texture
    like Minecraft blocks produces many ambiguous matches; this throws them out.
    """
    if des_query is None or des_train is None:
        return []
    # ORB descriptors are bit strings -> Hamming distance; SIFT are floats -> L2.
    norm = cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2
    knn = cv2.BFMatcher(norm).knnMatch(des_query, des_train, k=2)
    return [pair[0] for pair in knn
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]


# -------------------------------------------------------------- transforms --

def estimate_transform(kp_src, kp_dst, matches, model):
    """Find the 3x3 matrix mapping src frame pixels onto dst frame pixels.

    model:
      partial    - rotation + uniform scale + translation (4 DOF). Matches a
                   camera looking straight down at constant height. Default.
      affine     - also allows shear and non-uniform scale (6 DOF).
      homography - full perspective (8 DOF), needed only if the camera tilts.

    RANSAC fits the model to random subsets of matches and keeps the fit most
    matches agree with, so wrong matches (outliers) are ignored.
    Returns (matrix or None, number of inliers).
    """
    if len(matches) < 4:
        return None, 0
    src = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if model == "homography":
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    elif model == "affine":
        M, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                       ransacReprojThreshold=RANSAC_THRESHOLD)
    else:
        M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=RANSAC_THRESHOLD)
    if M is None:
        return None, 0
    if M.shape == (2, 3):
        # affine results are 2x3; add the [0 0 1] row so all models chain the same way
        M = np.vstack([M, [0, 0, 1]])
    return M, int(mask.sum())


def register_frames(frames, method, n_features, ratio, model):
    """Place every frame in frame 0's coordinates by chaining pairwise transforms.

    frames: list of (frame_index, image).
    Returns (placed, stats): placed = [(frame_index, image, 3x3 transform)],
    stats = one dict per attempted pair.

    If a pair fails, that frame is skipped and the next frame is matched against
    the last frame that was placed successfully.
    """
    detector = create_detector(method, n_features)
    ref_index, ref_image = frames[0]
    ref_kp, ref_des = detect_features(detector, ref_image)
    placed = [(ref_index, ref_image, np.eye(3))]
    stats = []

    for index, image in frames[1:]:
        kp, des = detect_features(detector, image)
        good = match_features(des, ref_des, method, ratio)
        M, inliers = estimate_transform(kp, ref_kp, good, model)
        ok = M is not None and inliers >= MIN_INLIERS
        stats.append({"pair": (ref_index, index), "keypoints": len(kp),
                      "good": len(good), "inliers": inliers, "ok": ok})
        if not ok:
            continue
        # M maps this frame -> reference frame; the reference's transform maps
        # reference -> frame 0. Multiplying gives this frame -> frame 0.
        placed.append((index, image, placed[-1][2] @ M))
        ref_index, ref_kp, ref_des = index, kp, des

    return placed, stats


# ------------------------------------------------------------------ canvas --

def compute_canvas(placed):
    """Return ((width, height), offset) big enough to hold every warped frame.

    offset is a translation that moves the top-left of the mosaic to (0, 0),
    since frames placed above or left of frame 0 have negative coordinates.
    """
    corners = []
    for _, image, T in placed:
        h, w = image.shape[:2]
        box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(box, T))
    pts = np.concatenate(corners).reshape(-1, 2)
    x_min, y_min = np.floor(pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(pts.max(axis=0)).astype(int)
    width, height = int(x_max - x_min), int(y_max - y_min)
    if max(width, height) > MAX_CANVAS_SIDE:
        raise RuntimeError(f"canvas would be {width}x{height}; transforms are "
                           "probably broken (try --model partial or a smaller -n)")
    offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    return (width, height), offset


def warp_and_overlay(placed, canvas_size, offset):
    """Warp each frame onto the canvas. Later frames are drawn over earlier ones."""
    width, height = canvas_size
    canvas = np.zeros((height, width, 3), np.uint8)
    for _, image, T in placed:
        H = offset @ T
        warped = cv2.warpPerspective(image, H, (width, height))
        # Warp a white mask the same way to know which canvas pixels this frame
        # covers. Only fully covered pixels (255) are copied, so the half-black
        # edge pixels from interpolation don't leave dark seams.
        mask = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), H,
                                   (width, height))
        covered = mask == 255
        canvas[covered] = warped[covered]
    return canvas


# ------------------------------------------------------------------ output --

def save_match_visualization(frames, method, n_features, ratio, path):
    """Draw the ratio-test matches between the first two frames, for inspection."""
    detector = create_detector(method, n_features)
    (_, img0), (_, img1) = frames[0], frames[1]
    kp0, des0 = detect_features(detector, img0)
    kp1, des1 = detect_features(detector, img1)
    good = match_features(des1, des0, method, ratio)
    vis = cv2.drawMatches(img1, kp1, img0, kp0, good[:100], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(path, vis)


def print_stats(stats, placed, frames, canvas_size, seconds):
    print(f"{'pair':>12} {'keypoints':>9} {'good':>6} {'inliers':>7}  status")
    for s in stats:
        pair = f"{s['pair'][0]}->{s['pair'][1]}"
        status = "ok" if s["ok"] else "FAILED"
        print(f"{pair:>12} {s['keypoints']:>9} {s['good']:>6} {s['inliers']:>7}  {status}")

    failures = sum(not s["ok"] for s in stats)
    good = [s["good"] for s in stats]
    inliers = [s["inliers"] for s in stats]
    print()
    print(f"frames sampled : {len(frames)}")
    print(f"frames used    : {len(placed)}")
    print(f"failed pairs   : {failures}")
    if stats:
        print(f"good matches   : mean {np.mean(good):.0f}, min {min(good)}")
        print(f"inliers        : mean {np.mean(inliers):.0f}, min {min(inliers)}")
    # Where the last frame ended up relative to frame 0. For a straight, level
    # flight, scale should stay ~1.0 and rotation ~0; anything else is drift.
    T = placed[-1][2]
    scale = np.sqrt(abs(np.linalg.det(T[:2, :2])))
    angle = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    print(f"last frame     : shift ({T[0, 2]:+.0f}, {T[1, 2]:+.0f}) px, "
          f"scale {scale:.3f}, rotation {angle:+.2f} deg")
    print(f"canvas         : {canvas_size[0]}x{canvas_size[1]}")
    print(f"time           : {seconds:.1f} s")


def main():
    parser = argparse.ArgumentParser(description="Stitch a top-down video into one image.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("-n", "--every-n", type=int, default=10,
                        help="keep one frame out of every N (default: 10)")
    parser.add_argument("-s", "--scale", type=float, default=0.5,
                        help="resize factor for frames (default: 0.5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="use at most this many frames")
    parser.add_argument("--method", choices=["orb", "sift"], default="orb",
                        help="feature detector (default: orb)")
    parser.add_argument("--features", type=int, default=3000,
                        help="max keypoints per frame (default: 3000)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Lowe ratio test threshold (default: 0.75)")
    parser.add_argument("--model", choices=["partial", "affine", "homography"],
                        default="partial", help="transform model (default: partial)")
    parser.add_argument("-o", "--output", default="results/stitched.jpg",
                        help="where to write the stitched image")
    parser.add_argument("--draw-matches", metavar="PATH",
                        help="also save a picture of the first pair's matches")
    args = parser.parse_args()

    start = time.time()
    frames = list(sample_frames(args.video, args.every_n, args.scale, args.limit))
    if len(frames) < 2:
        raise SystemExit("need at least 2 frames; lower -n or raise --limit")

    placed, stats = register_frames(frames, args.method, args.features,
                                    args.ratio, args.model)
    canvas_size, offset = compute_canvas(placed)
    mosaic = warp_and_overlay(placed, canvas_size, offset)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, mosaic)
    if args.draw_matches:
        save_match_visualization(frames, args.method, args.features, args.ratio,
                                 args.draw_matches)

    print_stats(stats, placed, frames, canvas_size, time.time() - start)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
