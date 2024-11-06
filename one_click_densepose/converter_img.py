import cv2
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser
import sys


# parser = ArgumentParser()
# parser.add_argument(
#     "--input", type=str, help="Set the input path to the video", required=True
# )
# parser.add_argument(
#     "--out", type=str, help="Set the output path to the video", required=True
# )
# args = parser.parse_args()

predictor = Predictor()
out_frame, out_frame_seg = predictor.predict(frame)

# Write the frame to the output video
# out.write(out_frame_seg)

# done += 1
# percent = int((done / n_frames) * 100)
# sys.stdout.write(
#     "\rProgress: [{}{}] {}%".format("=" * percent, " " * (100 - percent), percent)
# )
# sys.stdout.flush()
#
# Release video capture and writer
# cap.release()
# out.release()
