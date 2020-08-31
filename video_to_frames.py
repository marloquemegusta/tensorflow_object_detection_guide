import cv2
import argparse
import glob
import sys
import os

ap = argparse.ArgumentParser()

ap.add_argument("-o", "--output_directory", required=True,
                help="path to the output directory")
ap.add_argument("-d", "--decimate_factor", required=True, type=int,
                help="preserve 1 out of each d frames")
ap.add_argument("-i", "--input_video_path", required=True,
                help="path to the input video path")


def video_to_frames(path_to_video, output_dir, decimate_factor):
    if not glob.glob(path_to_video):
        sys.exit("input video does not exist")
    if not glob.glob(output_dir):
        os.makedirs(output_dir)
    vidcap = cv2.VideoCapture(path_to_video, cv2.CAP_FFMPEG)
    success = vidcap.grab()
    count = 0
    while success:
        if count % decimate_factor == 0:
            _, image = vidcap.retrieve()
            if not glob.glob(output_dir):
                sys.exit("output dir does not exists")
            cv2.imwrite(output_dir+"/"+path_to_video.split("/")[-1].split(".")[0]+"_frame%d.jpg" % count, image)
            print(output_dir+"/"+path_to_video.split("/")[-1].split(".")[0]+"_frame%d.jpg" % count, success)
        success = vidcap.grab()
        count += 1


if __name__ == "__main__":
    args = vars(ap.parse_args())
    video_to_frames(args["input_video_path"], args["output_directory"], args["decimate_factor"])
