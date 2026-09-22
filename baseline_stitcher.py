"""Step 3: baseline attempt using OpenCV's built-in high level Stitcher.

This is the "does the easy thing already work?" check, before writing our own
pipeline. SCANS mode assumes the images are related by affine transforms (a flat
scene photographed from roughly the same distance), which is the right model for
a camera looking straight down. PANORAMA mode assumes pure camera rotation about
a fixed point, which is not what a flying camera does.
"""

import argparse
import os

import cv2

from frame_extraction import sample_frames

# Stitcher.stitch() returns a status code; these are the possible values.
STATUS_NAMES = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS (too few usable matches)",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


def run_stitcher(frames, mode, confidence=None):
    """Run the built-in stitcher. Returns (status_code, panorama_or_None)."""
    stitcher = cv2.Stitcher.create(mode)
    if confidence is not None:
        # How confident the matcher must be that two images overlap before it
        # links them. Default is 1.0; lowering it accepts weaker matches.
        stitcher.setPanoConfidenceThresh(confidence)
    status, pano = stitcher.stitch(frames)
    return status, (pano if status == cv2.Stitcher_OK else None)


def main():
    parser = argparse.ArgumentParser(description="Baseline stitch with cv2.Stitcher.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("-n", "--every-n", type=int, default=10,
                        help="keep one frame out of every N (default: 10)")
    parser.add_argument("-s", "--scale", type=float, default=0.5,
                        help="resize factor for frames (default: 0.5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="use at most this many frames")
    parser.add_argument("--mode", choices=["scans", "panorama"], default="scans",
                        help="stitcher mode (default: scans)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="panorama confidence threshold, default 1.0 in OpenCV")
    parser.add_argument("-o", "--output", default="results/baseline_stitch.jpg",
                        help="where to write the panorama")
    args = parser.parse_args()

    frames = [f for _, f in sample_frames(args.video, args.every_n,
                                          args.scale, args.limit)]
    h, w = frames[0].shape[:2]
    print(f"loaded {len(frames)} frames at {w}x{h}")

    mode = cv2.Stitcher_SCANS if args.mode == "scans" else cv2.Stitcher_PANORAMA
    status, pano = run_stitcher(frames, mode, args.confidence)
    print(f"mode={args.mode} -> status {status}: {STATUS_NAMES.get(status, 'unknown')}")

    if pano is None:
        print("stitching failed, nothing written")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, pano)
    ph, pw = pano.shape[:2]
    print(f"wrote {args.output} ({pw}x{ph}, {pw / w:.2f}x wider and "
          f"{ph / h:.2f}x taller than one frame)")


if __name__ == "__main__":
    main()
