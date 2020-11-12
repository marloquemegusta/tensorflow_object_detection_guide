import numpy as np
import os
import glob
import argparse
import cv2
import json

# This script takes as an input a folder with the images we want to predict on. It fed this images to the specified
# model and the resulted predictions are printed as bounding boxes and save as images along with a json file containing
# the detections coordinates and if there is some part of the train rails missing. All of this while on the limits
# specified, and depending on the working mode
# It also creates an inform in txt about the erors comited during the prediction

# input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_path", required=True,
                help="path to the input model")
ap.add_argument("-o", "--output_directory", required=True,
                help="path to the output directory")
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("-i_gt", "--input_directory_gt", required=False, default = "",
                help="path to the input directory for the ground truth files")
ap.add_argument("-f", "--image_format", required=False, default="jpg",
                help="format of the images (jpg, JPG, png...)")
ap.add_argument("-t", "--threshold", required=False, default=0.9, type=float,
                help="threshold to draw a detection")
ap.add_argument("-l", "--labels", nargs="+", required=True)
ap.add_argument("-lim", "--limit", default=1, required=False, type=float,
                help="fraction of 1 of the image to keep, it will be crop starting from the bottom")


def analyse_single_img(image, gt, boxes,  labels, colors, threshold, limit):
    W = image.shape[1]
    H = image.shape[0]
    limit = limit*H
    clone = image.copy()
    data = {}
    id_to_label = {0: labels[0], 1: labels[1], 2: labels[2]}
    label_to_id = {labels[0]: 0, labels[1]: 1, labels[2]: 2}
    boxes[0, 0, :, 3:7] = boxes[0, 0, :, 3:7] * np.array([W, H, W, H])

    # if we have gt information, we compute its valid boxes
    if gt:
        boxes_gt = np.ones((1, 1, len(gt["shapes"]), 7))
        for i, shape in enumerate(gt["shapes"]):
            boxes_gt[0, 0, i, 1] = label_to_id[shape["label"]]
            boxes_gt[0, 0, i, 3:7] = np.array([shape["points"]]).flatten()

        cond1_gt = boxes_gt[0, 0, :, 2] > threshold
        cond2_gt = boxes_gt[0, 0, :, 6] < limit
        valid_boxes_gt = boxes_gt[0, 0][np.logical_and(cond1_gt, cond2_gt)]

        traviesas_gt = valid_boxes_gt[valid_boxes_gt[:, 1] == 0]
        primera_traviesa_gt = traviesas_gt[traviesas_gt[:, 6] == np.max(traviesas_gt[:, 6])][0]
        cond3_gt = valid_boxes_gt[:, 6] > primera_traviesa_gt[4]
        cond4_gt = valid_boxes_gt[:, 4] < primera_traviesa_gt[6]
        valid_boxes_gt = valid_boxes_gt[np.logical_and(cond3_gt, cond4_gt)]

    # computing the valid boxes according to the th and the specified limit
    cond1 = boxes[0, 0, :, 2] > threshold
    cond2 = boxes[0, 0, :, 6] < limit
    valid_boxes = boxes[0, 0][np.logical_and(cond1, cond2)]

    traviesas = valid_boxes[valid_boxes[:, 1] == 0]
    primera_traviesa = traviesas[traviesas[:, 6] == np.max(traviesas[:, 6])][0]
    cond3 = valid_boxes[:, 6] > primera_traviesa[4]
    cond4 = valid_boxes[:, 4] < primera_traviesa[6]
    valid_boxes = valid_boxes[np.logical_and(cond3, cond4)]

    # draw the selected boxes
    for box_to_draw in valid_boxes:
        color = colors[int(box_to_draw[1])]
        color = color.tolist()[0], color.tolist()[1], color.tolist()[2]
        cv2.rectangle(clone, (int(box_to_draw[3]), int(box_to_draw[4])),
                      (int(box_to_draw[5]), int(box_to_draw[6])), color, 10)
        text = labels[int(box_to_draw[1])]
        cv2.putText(clone, text, (int(box_to_draw[3]), int(box_to_draw[4]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # create json file with the detections and the missing elements
    data["detections"] = list(range(0, len(valid_boxes)))
    recuento_traviesas = 0
    recuento_clips = 0
    recuento_tornillos = 0
    for i in range(valid_boxes.shape[0]):
        label = id_to_label[valid_boxes[i, 1]]
        valid_boxes[i][3:7] = valid_boxes[i][3:7] * np.array([W, H, W, H])
        upper_left = (float(valid_boxes[i, 3]), float(valid_boxes[i, 4]))
        bottom_right = (float(valid_boxes[i, 5]), float(valid_boxes[i, 6]))
        data["detections"][i] = {"label": label, "points": [upper_left, bottom_right]}
        recuento_traviesas = recuento_traviesas + 1 if label == "traviesa" else recuento_traviesas
        recuento_clips = recuento_clips + 1 if label == "clip" else recuento_clips
        recuento_tornillos = recuento_tornillos + 1 if label == "tornillo" else recuento_tornillos
        data["incidencias"] = {"traviesa": 1 - recuento_traviesas, "clip": 2 - recuento_clips,
                               "tornillos": 2 - recuento_tornillos}

    # return number of fp and fn depending on if we have gt information
    failures = [sum(valid_boxes_gt[:, 1] == i) - sum(valid_boxes[:, 1] == i) for k in range(len(labels))] if gt else []
    return clone, data, np.array(failures)


if __name__ == "__main__":
    args = vars(ap.parse_args())
    if not glob.glob(args["output_directory"]):
        os.makedirs(args["output_directory"])
        print("output directory doesn't exist, creating it")

    json_paths = np.array(glob.glob(args["input_directory_gt"] + "/*." + "json"))
    ntotal_files = len(json_paths)
    image_paths = np.array(glob.glob(args["input_directory"] + "/*." + args["image_format"]))
    cvNet = cv2.dnn.readNetFromTensorflow(args["model_path"] + '/frozen_inference_graph.pb',
                                          args["model_path"] + '/graph.pbtxt')
    num_classes = len(args["labels"])
    available_colors = np.random.randint(0, 255, (num_classes, 3))
    imagenes_con_fallo = []
    false_positives = 0
    num_incidencias = 0
    imagenes_con_incidencia = []

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        json_data = json.load(open(json_paths[i])) if json_paths else {}

        cvNet.setInput(cv2.dnn.blobFromImage(img, size=(1000, 1000), swapRB=True, crop=False))
        b = cvNet.forward()
        result = analyse_single_img(image=img,
                                    gt=json_data,
                                    boxes=b,
                                    labels=args["labels"],
                                    threshold=args["threshold"],
                                    colors=available_colors,
                                    limit=args["limit"])

        image_data = result[0]
        json_data = result[1]

        if sum(result[2] != 0) > 0:
            imagenes_con_fallo.append(image_path.split(os.sep)[-1].split(".")[0])
            false_positives += sum(np.abs(result[2][result[2] < 0]))

        for elemento in json_data["incidencias"] :
            if json_data["incidencias"][elemento] != 0:
                num_incidencias += json_data["incidencias"][elemento]
                imagenes_con_incidencia.append(image_path)

        # save image and json data
        cv2.imwrite(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg",
                    image_data)
        json_data["image_path"] = image_path
        with open(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + ".json",
                  'w') as outfile:
            json.dump(json_data, outfile, indent=4)
        print(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg")

    # write a report
    file = open("report_incidences.txt", "w")
    str1 = "El número de incidencias es : {} \n".format(num_incidencias)
    str2 = "imágenes analizadas : {} \n Imágenes con incidencia: \n".format(ntotal_files)
    str3 = '\n '.join(map(str, imagenes_con_incidencia))
    file.writelines([str1, str2, str3])
    file.close()

    if result[2].shape[0] != 0:
        print(imagenes_con_fallo)
        # write a report
        file = open("report_model_performance.txt", "w")
        str1 = "El número de falsos postivos en este conjunto de imágenes de test es : {} \n".format(false_positives)
        str2 = "imágenes analizadas : {} \n".format(ntotal_files)
        file.writelines([str1, str2])
        file.close()
        print("tenemos un total de {}  FP, que suponen un {}".format(false_positives,
                                                                     100 * len(imagenes_con_fallo) / ntotal_files)," %")