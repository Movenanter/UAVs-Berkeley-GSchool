"""Step 2: split a video into frames, keeping only every Nth one.

Used as a library by the stitching scripts, and runnable on its own to dump
sample frames to disk so you can eyeball them.
"""

import argparse
import os

import cv2


def sample_frames(video_path, every_n=5, scale=1.0, limit=None):
    """Yield (frame_index, image) for every Nth frame of the video.

    every_n : keep frame i only when i % every_n == 0
    scale   : resize factor applied to each kept frame (0.5 = half size)
    limit   : stop after this many kept frames (None = whole video)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    frame_index = 0
    kept = 0
    while True:
        # read() decodes the next frame; ok is False at end of file
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % every_n == 0:
            if scale != 1.0:
                # INTER_AREA is the right filter for shrinking an image
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            yield frame_index, frame
            kept += 1
            if limit is not None and kept >= limit:
                break

        frame_index += 1

    cap.release()


def video_info(video_path):
    """Return (frame_count, fps, width, height) without decoding the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")
    info = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def main():
    parser = argparse.ArgumentParser(description="Extract every Nth frame of a video.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("-n", "--every-n", type=int, default=5,
                        help="keep one frame out of every N (default: 5)")
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                        help="resize factor for saved frames (default: 1.0)")
    parser.add_argument("-o", "--out-dir", default="frames",
                        help="directory to write frames into (default: frames)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many saved frames")
    args = parser.parse_args()

    count, fps, width, height = video_info(args.video)
    print(f"video: {width}x{height}, {fps:.2f} fps, {count} frames")
    print(f"keeping every {args.every_n}th frame -> about {count // args.every_n} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    saved = 0
    size = None
    for frame_index, frame in sample_frames(args.video, args.every_n,
                                            args.scale, args.limit):
        path = os.path.join(args.out_dir, f"frame_{frame_index:05d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
        size = frame.shape[1], frame.shape[0]

    print(f"saved {saved} frames of size {size} to {args.out_dir}/")


if __name__ == "__main__":
    main()
