import numpy as np
import os
import glob
import argparse
import cv2

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
ap.add_argument("-eb", "--exclude_boxes", nargs="+", default=[], required=False,
                help="don't show boxes for detections with these labels")
ap.add_argument("-em", "--exclude_masks", nargs="+", default=[], required=False,
                help="don't show masks for detections with these labels")


def visualize_single_img(image, boxes, exclude_mask, exclude_box, colors, threshold, masks=None):
    W = image.shape[1]
    H = image.shape[0]
    clone = image.copy()
    class_mask = np.zeros(image.shape).astype("uint8")
    boxes_to_draw = []
    for i in range(boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        if confidence > threshold:
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            color = colors[classID]
            if (classID not in exclude_mask) and (masks is not None):
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH),
                                  interpolation=cv2.INTER_CUBIC)
                mask = (mask > threshold)
                class_mask[startY:endY, startX:endX][mask] = color
                # store the blended ROI in the original image
            if classID not in exclude_box:
                boxes_to_draw.append(((startX, startY), (endX, endY), classID))
    clone[class_mask != (0, 0, 0)] = clone[class_mask != (0, 0, 0)] * 0.4 + class_mask[class_mask != (0, 0, 0)] * 0.6
    for box_to_draw in boxes_to_draw:
        color = colors[box_to_draw[2]]
        color = color.tolist()[0], color.tolist()[1], color.tolist()[2]
        cv2.rectangle(clone, box_to_draw[0], box_to_draw[1], color, 3)
        text = LABELS[box_to_draw[2]]
        cv2.putText(clone, text, (box_to_draw[0][0], box_to_draw[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return clone


if __name__ == "__main__":
    args = vars(ap.parse_args())
    if not glob.glob(args["output_directory"]):
        os.makedirs(args["output_directory"])
        print("output directory doesn't exist, creating it")

    image_paths = np.array(glob.glob(args["input_directory"] + "/*." + args["image_format"]))
    cvNet = cv2.dnn.readNetFromTensorflow(args["model_path"] + '/frozen_inference_graph.pb',
                                          args["model_path"] + '/graph.pbtxt')
    LABELS = args["labels"]
    num_classes = len(LABELS)
    exclude_mask = args["exclude_masks"]
    exclude_box = args["exclude_boxes"]
    colors = np.random.randint(0, 255, (num_classes, 3))
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if "mask" in args["model_path"]:
            cvNet.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))
            (boxes, masks) = cvNet.forward(["detection_out_final", "detection_masks"])
            result = visualize_single_img(img, boxes, exclude_mask, exclude_box, colors, args["threshold"], masks=masks)
        else:
            cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
            boxes = cvNet.forward()
            result = visualize_single_img(img, boxes, exclude_mask, exclude_box, colors, args["threshold"])

        cv2.imwrite(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg",
                    result)
        print(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg")
