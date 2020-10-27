import json
import numpy as np
from lxml import etree as ET
import argparse
import glob
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_dir", required=True,
                help="path to the input directory")
ap.add_argument("-o", "--output_dir", required=True,
                help="output directory")


def json_to_xml(path_to_json):
    data = json.load(open(path_to_json))
    # detections is a list that contains one array for each detected object. Every array will contain another array with
    # one element per point that describes the polygon that masks the object. Every element contains 2 coordinates as
    # (x,y)
    labels = list(map(lambda x: x["label"], data["shapes"]))
    detections = list(map(lambda x: np.array(x["points"]), data["shapes"]))
    bb_list = []
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = "images"
    filename_ = ET.SubElement(annotation, 'filename')
    filename_.text = data["imagePath"]
    path = ET.SubElement(annotation, 'path')
    path.text = data["imagePath"]
    source = ET.SubElement(annotation, 'source')
    source.text = "yo mismo"
    database = ET.SubElement(source, 'database')
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(data["imageWidth"])
    height = ET.SubElement(size, 'height')
    height.text = str(data["imageHeight"])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = str(0)
    c = 0
    # we iterate over the detections list
    for detection in detections:
        # if the json file contains only bounding boxes it will only contain 2 points for each detection (upper left
        # and lower right corners of the bounding box). If so, we need to extract the other 2 edges of the bounding box
        if detection.shape[0] == 2:
            detection = np.array([detection[0], [detection[0][0], detection[1][1]], detection[1],
                                  [detection[1][0], detection[0][1]]]).astype(int)
        bbmin = int(min(detection[:, 0])), int(min(detection[:, 1]))
        bbmax = int(max(detection[:, 0])), int(max(detection[:, 1]))
        object_ = ET.SubElement(annotation, 'object')

        name = ET.SubElement(object_, 'name')
        name.text = labels[c]

        pose = ET.SubElement(object_, 'pose')
        pose.text = "Unspecified"

        truncated = ET.SubElement(object_, 'truncated')
        truncated.text = str(0)

        difficult = ET.SubElement(object_, 'difficult')
        difficult.text = str(0)

        bndbox = ET.SubElement(object_, 'bndbox')

        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bbmin[0])

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bbmin[1])

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bbmax[0])

        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bbmax[1])
        c += 1
    tree = ET.ElementTree(annotation)
    return tree


if __name__ == "__main__":
    args = vars(ap.parse_args())
    if not glob.glob(args["input_dir"]):
        sys.exit("input dir does not exist")
    json_paths = np.array(glob.glob(args["input_dir"] + "/*.json"))
    for json_path in json_paths:
        tree = json_to_xml(json_path)
        tree.write(args["output_dir"]+"/"+tree.find("filename").text.split(".")[0]+".xml", pretty_print=True)
        print(tree.find("filename").text.split(".")[0]+".xml")