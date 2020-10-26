import glob
import os
import json
import argparse
import sys

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_dir", required=True,
                help="input dir")
ap.add_argument("-ol", "--old_label", required=True, action="append",
                help="old label to be changed")
ap.add_argument("-nl", "--new_label", required=True, action="append",
                help="new changed label")


def rename_label(json_path, old_labels, new_labels):
    if not old_labels:
        sys.exit("not old labels")
    if len(old_labels) != len(new_labels):
        sys.exit("old labels and new labels don't have same length")

    json_data = json.load(open(json_path))
    for shape in json_data["shapes"]:
        for i, old_label in enumerate(old_labels):
            if shape["label"] == old_label:
                shape["label"] = new_labels[i]
    return json_data


if __name__ == "__main__":
    args = vars(ap.parse_args())
    json_paths = glob.glob(args["input_dir"] + "/*.json")
    if not json_paths:
        sys.exit("invalid input path")
    for json_path in json_paths:
        data = rename_label(json_path, args["old_label"], args["new_label"])
        with open(json_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)
