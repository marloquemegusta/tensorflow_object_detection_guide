import json
import pandas as pd
import argparse
import os
import glob
import numpy as np
import sys
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("-o", "--output_path", required=True,
                help="path to the output directory")


def json_to_csv(path_to_json):
    data = json.load(open(path_to_json))
    filename = data["imagePath"]
    width, height = data["imageWidth"], data["imageHeight"]
    labels = list(map(lambda x: x["label"], data["shapes"]))
    points = list(map(lambda x: np.array(x["points"]), data["shapes"]))
    c = 0
    df = pd.DataFrame()
    # we iterate over the labels looking for the 2 points that define the box
    for label in labels:
        d = {"filename": filename, "width": width, "height": height, "class": label,
             "xmin": min(points[c][:, 0]), "ymin": min(points[c][:, 1]),
             "xmax": max(points[c][:, 0]), "ymax": max(points[c][:, 1])}
        df = df.append(d, ignore_index=True)
        df = df[['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
        df = df.astype({'xmin': 'int32', 'xmax': 'int32', 'ymin': 'int32', 'ymax': 'int32'})
        c += 1
    return df


if __name__ == "__main__":
    args = vars(ap.parse_args())
    if not glob.glob(args["input_dir"]):
        sys.exit("input dir does not exist")
    jsonPaths = np.array(glob.glob(args["input_directory"]+"/*.json"))
    df = pd.DataFrame()
    c = 0
    for jsonPath in jsonPaths:
        df = df.append(json_to_csv(jsonPath))
        print(c)
        c += 1
    df.to_csv(args["output_path"],index=False)
    print("successfully converted json files to csv")
