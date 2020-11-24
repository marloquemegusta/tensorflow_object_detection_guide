import numpy as np
import os
import glob
import argparse
import cv2
import utilities

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_path", required=True,
                help="path to the input model")
ap.add_argument("-o", "--output_directory", required=True,
                help="path to the output directory")
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("-f", "--image_format", required=False, default="jpg",
                help="format of the images (jpg, JPG, png...)")
ap.add_argument("-t", "--threshold", required=False, default=0.9, type=float,
                help="threshold to draw a detection")
ap.add_argument("-l", "--labels", nargs="+", required=True)
ap.add_argument("-eb", "--exclude_boxes", nargs="+", default=[], required=False, type=int,
                help="don't show boxes for detections with these labels")
ap.add_argument("-em", "--exclude_masks", nargs="+", default=[], required=False, type=int,
                help="don't show masks for detections with these labels")


if __name__ == "__main__":
    args = vars(ap.parse_args())
    if not glob.glob(args["output_directory"]):
        os.makedirs(args["output_directory"])
        print("output directory doesn't exist, creating it")

    image_paths = np.array(glob.glob(args["input_directory"] + "/*." + args["image_format"]))
    cvNet = cv2.dnn.readNetFromTensorflow(args["model_path"] + '/frozen_inference_graph.pb',
                                          args["model_path"] + '/graph.pbtxt')
    num_classes = len(args["labels"])
    available_colors=np.random.randint(0, 255, (num_classes, 3))
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if "mask" in args["model_path"]:
            b,m = utilities.predict_on_single_image(img, cvNet,masks=True)
            result=utilities.visualize_predictions(image=img,
                                                   colors=available_colors,
                                                   labels=args["labels"],
                                                   boxes=b,
                                                   masks=m,
                                                   exclude_masks=args["exclude_masks"],
                                                   exclude_boxes=args["exclude_boxes"],
                                                   threshold=args["threshold"])
        else:
            b= utilities.predict_on_single_image(img, cvNet, masks=False)
            result = utilities.visualize_predictions(image=img,
                                                     colors=available_colors,
                                                     labels=args["labels"],
                                                     boxes=b,
                                                     masks=None,
                                                     exclude_masks=args["exclude_masks"],
                                                     exclude_boxes=args["exclude_boxes"],
                                                     threshold=args["threshold"])

        cv2.imwrite(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg",
                    result)
        print(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg")
