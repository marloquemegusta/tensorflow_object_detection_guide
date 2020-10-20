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
ap.add_argument("-l", "--path_to_labels", required=False,
                help="path to the labels.pbtxt file")
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("-f", "--image_format", required=False, default="jpg",
                help="format of the images (jpg, JPG, png...")
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("-t", "--threshold", required=False, default="0.9",
                help="threshold to draw a detection")


def visualize_single_img(image, exclude_mask, exclude_box, colors, threshold=0.9):
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
            if classID not in exclude_mask:
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
    image_paths = np.array(glob.glob(args["input_directory"] + "/*." + args["image_format"]))
    cvNet = cv2.dnn.readNetFromTensorflow(args["model_path"] + '/frozen_inference_graph.pb',
                                          args["model_path"] + '/graph.pbtxt')
    LABELS = ["via", "catenaria", "pk", "senal fija", "senal luminosa"]
    num_classes = 5
    exclude_mask = [1, 2, 3, 4]
    exclude_box = [0]
    colors = np.random.randint(0, 255, (num_classes, 3))
    for image_path in image_paths:
        img = cv2.imread(image_path)
        cvNet.setInput(cv2.dnn.blobFromImage(img, swapRB=True, crop=False))
        (boxes, masks) = cvNet.forward(["detection_out_final", "detection_masks"])
        W = img.shape[1]
        H = img.shape[0]
        result = visualize_single_img(img, exclude_mask, exclude_box, colors, args["threshold"])
        cv2.imwrite(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg",
                    result)
        print(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg")
