import os
import argparse
import glob
from shutil import copyfile
import numpy as np
import random

# File to pick the desired number of images from the original images

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output_directory", required=True,
                help="path to the output directory")
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("-df", "--datasetfinal_directory", required=False, default="",
                help="path to the datasetfinal directory")
ap.add_argument("-f", "--image_format", required=False, default="JPG",
                help="format of the images (jpg, JPG, png...)")
ap.add_argument("-n", "--num_img", required=True, type=int,
                help="number of desired output images")

args = vars(ap.parse_args())

# if there the output directory does not exit, create it
if not glob.glob(args["output_directory"]):
    os.makedirs(args["output_directory"])
    print("output directory doesn't exist, creating it")

# to check if the images already exists in the dataset final directory
# if the dataset final directory does not exist, we create an empty list
if not args["datasetfinal_directory"]: # if it is empty, this will return true
    image_paths_df = []
else:
    image_paths_df = np.array(glob.glob(args["datasetfinal_directory"] + "/*." + args["image_format"]))
    
# create the image paths from the input directory
image_paths = np.array(glob.glob(args["input_directory"] + "/*." + args["image_format"]))

names = list(map(lambda x: x.split(os.sep)[-1].split(".")[0], image_paths))
names_df = list(map(lambda x: x.split(os.sep)[-1].split(".")[0], image_paths_df))
names = list(filter(lambda x: x not in names_df, names))
random.shuffle(names)
keep_names = names[0:args["num_img"]]

for name in keep_names:
    if name not in names_df:
        print(name)
        copyfile("{}/{}.{}".format(args["input_directory"], name, args["image_format"]),
                 "{}/{}.{}".format(args["output_directory"], name, args["image_format"]))

print("succesfully split ", args["num_img"]," images from the total of ", image_paths.shape[0])
