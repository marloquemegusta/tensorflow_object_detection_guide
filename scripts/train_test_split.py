import numpy as np
import argparse
import glob
import sys
import os
from shutil import copyfile

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_dir", required=True,

                help="path to the input directory")
ap.add_argument("-o", "--output_dir", required=True,
                help="output directory")
ap.add_argument("-vr", "--validation_rate", required=True, type=float,
                help="fraction of the data to be used as validation")

ap.add_argument("-tr", "--test_rate", required=True, type=float,
                help="fraction of the data to be used as test")

ap.add_argument("-f", "--image_format", required=True, type=str,
                help="format o the image (jpg,png,JPG...")


def partition_dataset(input_dir, output_dir, test_rate, val_rate,image_format="jpg"):
    if not glob.glob(input_dir):
        sys.exit("input dir does not exist")
    if not glob.glob(output_dir):
        sys.exit("output dir does not exist")
    if not glob.glob(output_dir + "/test") and test_rate > 0:
        os.makedirs(output_dir + "/test")
    if not glob.glob(output_dir + "/train"):
        os.makedirs(output_dir + "/train")
    if not glob.glob(output_dir + "/val") and val_rate > 0:
        os.makedirs(output_dir + "/val")

    json_paths = np.array(glob.glob(input_dir + "/*.json"))
    np.random.shuffle(json_paths)
    names = list(map(lambda x: x.split(os.sep)[-1].split(".")[0], json_paths))
    folder_name = input_dir.split("/")[-1]
    val_index = round(json_paths.shape[0] * val_rate)
    test_index = round(json_paths.shape[0] * val_rate) + round(json_paths.shape[0] * test_rate)
    val_names, test_names, train_names = np.split(names, [val_index, test_index])

    for val_name in val_names:
        copyfile(input_dir + "/" + val_name + ".json",
                 output_dir + "/val/" + val_name + ".json")
        copyfile(input_dir + "/" + val_name + "."+image_format,
                 output_dir + "/val/" + val_name + "."+image_format)
    print("succesfully created validation split using: ", val_rate*100,"% of data")
    for test_name in test_names:
        copyfile(input_dir + "/" + test_name + ".json",
                 output_dir + "/test/" + test_name + ".json")
        copyfile(input_dir + "/" + test_name + "."+image_format,
                 output_dir + "/test/" + test_name + "."+image_format)
    print("succesfully created test split using: ", test_rate*100,"% of data")
    for train_name in train_names:
        copyfile(input_dir + "/" + train_name + ".json",
                 output_dir + "/train/" + train_name + ".json")
        copyfile(input_dir + "/" + train_name + "."+image_format,
                 output_dir + "/train/" + train_name + "."+image_format)
    print("succesfully created train split using: ", (1-test_rate-val_rate)*100,"% of data")

if __name__ == "__main__":
    args = vars(ap.parse_args())
    partition_dataset(args["input_dir"], args["output_dir"], args["test_rate"],args["validation_rate"],args["image_format"])
