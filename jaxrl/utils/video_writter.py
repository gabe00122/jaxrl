import numpy as np

# import ffmpeg
from ffmpeg import FFmpeg


# converts video grayscale to rgb
def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    return image


# def save_frame(frame: np.ndarray):
#     import matplotlib.pyplot as plt
#     # save to a file
#     plt.imshow(frame)
#     plt.savefig('frame.png')
#     plt.close()


def save_video(frames: np.ndarray, filename, fps=60):
    print("Saving video...")
    # frames = grayscale_to_rgb(frames)

    h, w = frames[0].shape[:2]
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(
            "pipe:0",
            {
                "f": "rawvideo",
                "pix_fmt": "rgb24",
                "s": f"{w}x{h}",
                "framerate": fps,
            },
        )
        .output(
            filename,
            {
                "vf": "scale=iw*4:ih*4:flags=neighbor",
                "pix_fmt": "yuv420p",
                "codec:v": "libx264",
                "movflags": "faststart",
            },
        )
    )
    print(" ".join(ffmpeg.arguments))

    vid_bytes = frames.astype(np.uint8).tobytes()

    try:
        output = ffmpeg.execute(vid_bytes, 1000)
        print(output.decode("utf-8"))
    except Exception as e:
        print(e)
