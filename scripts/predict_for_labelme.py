import cv2
import json
import numpy as np
import io
from PIL import Image
import base64
import os
import argparse
import glob
from shutil import copyfile

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_path", required=True,
                help="path to the input model")
ap.add_argument("-o", "--output_directory", required=True,
                help="path to the output directory")
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("-f", "--image_format", required=False, default="JPG",
                help="format of the images (jpg, JPG, png...)")
ap.add_argument("-t", "--threshold", required=False, default=0.9, type=float,
                help="threshold to draw a detection")
ap.add_argument("-l", "--labels", nargs="+", default=[], required=True)


def predict_and_generate_labelme_json(image, labels, boxes, threshold=0.5):
    W = image.shape[1]
    H = image.shape[0]
    data = {}
    labels_id = {0: labels[0], 1: labels[1], 2: labels[2]}
    data["version"] = "4.5.5"
    data["flags"] = {}
    valid_boxes = boxes[0, 0][boxes[0, 0, :, 2] > threshold]
    data["shapes"] = list(range(0, len(valid_boxes)))
    for i in range(valid_boxes.shape[0]):
        label = labels_id[valid_boxes[i, 1]]
        valid_boxes[i][3:7] = valid_boxes[i][3:7] * np.array([W, H, W, H])
        upper_left = (float(valid_boxes[i, 3]), float(valid_boxes[i, 4]))
        bottom_right = (float(valid_boxes[i, 5]), float(valid_boxes[i, 6]))
        data["shapes"][i] = {"label": label, "points": [upper_left, bottom_right], "group_id": None,
                             "shape_type": "rectangle",
                             "flags": {}}

    pil_image = Image.fromarray(image[:, :, ::-1])
    rawBytes = io.BytesIO()
    pil_image.save(rawBytes, "JPEG")
    # return to the start of the file
    encoded = base64.b64encode(rawBytes.getvalue()).decode("utf-8")
    data["imagePath"] = image_path.split(os.sep)[-1]
    data["imageData"] = encoded
    data["imageHeight"] = H
    data["imageWidth"] = W
    return data


if __name__ == "__main__":
    args = vars(ap.parse_args())
    if not glob.glob(args["output_directory"]):
        os.makedirs(args["output_directory"])
        print("output directory doesn't exist, creating it")

    image_paths = np.array(glob.glob(args["input_directory"] + "/*." + args["image_format"]))
    print("{} images found".format(len(image_paths)))
    cvNet = cv2.dnn.readNetFromTensorflow(args["model_path"] + '/frozen_inference_graph.pb',
                                          args["model_path"] + '/graph.pbtxt')
    LABELS = args["labels"]
    print("{} will be used as labels".format(LABELS))
    num_classes = len(LABELS)
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if "mask" in args["model_path"]:
            # cvNet.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))
            # result,_ = cvNet.forward(["detection_out_final", "detection_masks"])
            # json_data = predict_and_generate_labelme_json(img, boxes, args["labels"], args["threshold"])
            print("not working with masks yet")
        else:
            cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
            result = cvNet.forward()
            #print("predictions for {} created ".format(image_path))
            json_data = predict_and_generate_labelme_json(img, LABELS, result, args["threshold"])

        copyfile(image_path, args["output_directory"] + "/" + image_path.split(os.sep)[-1])
        with open(args["output_directory"] + "/" + image_path.split(os.sep)[-1].replace(args["image_format"], "json"),
                  'w') as outfile:
            json.dump(json_data, outfile, indent=4)
        print("json for {} created ".format(image_path))
