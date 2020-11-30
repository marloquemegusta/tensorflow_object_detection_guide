import numpy as np
import os
import glob
import argparse
import cv2
import json
import utilities

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
ap.add_argument("-f", "--image_format", required=False, default="JPG",
                help="format of the images (jpg, JPG, png...)")
ap.add_argument("-t", "--threshold", required=False, default=0.9, type=float,
                help="threshold to draw a detection")
ap.add_argument("-l", "--labels", nargs="+", required=True)
ap.add_argument("-lim", "--limit", default=1, required=False, type=float,
                help="fraction of 1 of the image to keep, it will be crop starting from the bottom")


if __name__ == "__main__":
    args = vars(ap.parse_args())
    if not glob.glob(args["output_directory"]):
        os.makedirs(args["output_directory"])
        print("output directory doesn't exist, creating it")

    image_paths = np.array(glob.glob(args["input_directory"] + "/*." + args["image_format"]))
    cvNet = cv2.dnn.readNetFromTensorflow(args["model_path"] + '/frozen_inference_graph.pb',
                                          args["model_path"] + '/graph.pbtxt')
    num_classes = len(args["labels"])
    ntotal_files=len(image_paths)
    available_colors = np.random.randint(0, 255, (num_classes, 3))
    num_incidencias = 0
    imagenes_con_incidencia = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        detections = utilities.predict_on_single_image(img,cvNet)
        first_traviesa_detections = utilities.pick_largest_traviesa(detections, args["threshold"])
        if first_traviesa_detections is not None:
            image_with_detections=utilities.visualize_predictions(image=img,
                                                                  colors=available_colors,
                                                                  labels=args["labels"],
                                                                  boxes=first_traviesa_detections,
                                                                  threshold=args["threshold"])
            incidence_data = utilities.generate_incidence(boxes=first_traviesa_detections,
                                                          W=img.shape[1],
                                                          H=img.shape[0],
                                                          labels=args["labels"],
                                                          )
            for campo in incidence_data["incidencias"] :
                if incidence_data["incidencias"][campo] != 0:
                    num_incidencias += incidence_data["incidencias"][campo]
                    imagenes_con_incidencia.append(image_path)
            # save image and json data
            cv2.imwrite(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg",
                        image_with_detections)
            incidence_data["original_image_path"] = image_path
            incidence_data["processed_image_path"] = args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg"
            with open(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + ".json",
                      'w') as outfile:
                json.dump(incidence_data, outfile, indent=4)
            print(args["output_directory"] + "/" + image_path.split(os.sep)[-1].split(".")[0] + "_predictions.jpg")
        else:
            print("no se ha encontrado traviesa en la imagen {}".format(image_path))

    # write a report
    file = open(args["output_directory"] + "/report_incidences.txt", "w")
    str1 = "El número de incidencias es : {} \n".format(num_incidencias)
    str2 = "imágenes analizadas : {} \n Imágenes con incidencia: \n".format(ntotal_files)
    str3 = '\n '.join(map(str, imagenes_con_incidencia))
    file.writelines([str1, str2, str3])
    file.close()