import glob
import os
import io
from PIL import Image
import cv2
import base64
import argparse
import sys
import json

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="input dir")
ap.add_argument("-o", "--output_dir", required=True,
                help="output directory")
ap.add_argument("-r", "--resize_ratio", required=True, type=int,
                help="resize ratio as a division factor")


def resize_json_and_img(image_path, resize_ratio):
    json_path = image_path.replace("JPG", "json")
    json_data = json.load(open(json_path))
    img_array = cv2.imread(image_path)
    target_size = [int(dimension / resize_ratio) for dimension in img_array.shape[0:2]][::-1]
    target_size = tuple(target_size)
    json_data["imageHeight"] = target_size[0]
    json_data["imageWidth"] = target_size[1]
    img_array = cv2.resize(img_array, target_size)
    img = Image.fromarray(img_array[:,:,::-1])
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    # return to the start of the file
    encoded = base64.b64encode(rawBytes.getvalue()).decode("utf-8")
    json_data["imageData"] = encoded
    for shape in json_data["shapes"]:
        shape["points"] = [[coord / resize_ratio for coord in point] for point in shape["points"]]
    return json_data, img_array


if __name__ == "__main__":
    args = vars(ap.parse_args())
    image_paths = glob.glob(args["input_dir"] + "/*.jpg")
    if not image_paths:
        sys.exit("invalid input path")
    if not glob.glob(args["output_dir"]):
        os.makedirs(args["output_dir"])
        print("output directory does not exist, creating it...")
    for image_path in image_paths:
        data, img = resize_json_and_img(image_path, args["resize_ratio"])
        with open(args["output_dir"]+"/"+image_path.split(os.sep)[-1].replace("JPG", "json"), 'w') as outfile:
            json.dump(data, outfile, indent=4)
        cv2.imwrite(args["output_dir"]+"/"+image_path.split(os.sep)[-1], img)
        print(args["output_dir"]+"/"+image_path.split(os.sep)[-1])
