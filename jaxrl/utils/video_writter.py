import numpy as np
# import ffmpeg
from ffmpeg import FFmpeg

# converts video grayscale to rgb
def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:  # If frames are grayscale (num_frames, height, width)
        image = np.stack((image,) * 3, axis=-1)  # Convert to RGB by stacking
    return image

# def save_frame(frame: np.ndarray):
#     import matplotlib.pyplot as plt
#     # save to a file
#     plt.imshow(frame)
#     plt.savefig('frame.png')
#     plt.close()

def save_video(frames: np.ndarray, filename, fps=60):
    print("Saving video...")
    frames = grayscale_to_rgb(frames)

    h, w = frames[0].shape[:2]
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input("pipe:0", {"f": "rawvideo", "pix_fmt": "rgb24", "s": f"{w}x{h}"})
        .output(filename, {"vf": "scale=iw*4:ih*4:flags=neighbor", "pix_fmt": "yuv420p", "codec:v": "libx264", "framerate": fps})
    )
    print(' '.join(ffmpeg.arguments))

    vid_bytes = frames.astype(np.uint8).tobytes()

    try:
        output = ffmpeg.execute(vid_bytes, 1000)
        print(output.decode("utf-8"))
    except Exception as e:
        print(e)


class OldVideoWriter:
    def __init__(
        self,
        fn,
        vcodec="libx264",
        fps=60,
        in_pix_fmt="rgb24",
        out_pix_fmt="yuv420p",
        input_args=None,
        output_args=None,
    ):
        self.fn = fn
        self.process = None
        self.input_args = {} if input_args is None else input_args
        self.output_args = {} if output_args is None else output_args
        self.input_args["framerate"] = fps
        self.input_args["pix_fmt"] = in_pix_fmt
        self.output_args["pix_fmt"] = out_pix_fmt
        self.output_args["vcodec"] = vcodec

    def add(self, frame):
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s=f"{w}x{h}",
                    **self.input_args
                )
                .output("output.mp4", **self.output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()
