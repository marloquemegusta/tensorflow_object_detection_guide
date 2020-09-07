# Tensorflow object detection API guide

This document aims to cover the whole process of training an object detection model using tensorflow object detection API. Their API is well documented, but there are some things that are not so easy to follow. This guide asumes you alredy have your images labeled using labelme tool available here: https://github.com/wkentaro/labelme.
We will review:
  - How to install the API itself and how to troubleshoot some o the problems that you could encounter (because I did)
  - How to convert your annotations file (.json files) produced by labeleme and turn them into .tfrecord files, which are required to train a tensorflow model. We will review the process for bounding boxes only annotations and mask annotations also
  - How to train the model by downloading a pre-trained one from tensorflow model-zoo and fine tunning it using the parameters set in a config file
  - How to export the trained model and use it to generate predictions on a given set of images
    

## Cloning this repo
You need to clone this repository, as this guide asumes you follow the same directory sctructure that I do:
![project structure](https://github.com/marloquemegusta/tensorflow_object_detection_guide/blob/master/project_structure.PNG?raw=true)
You can do it by typing:
```sh
$ git clone https://github.com/marloquemegusta/tensorflow_object_detection_guide.git
```

##  Setting up the environment and all the libraries and files needed
In this guide we will be using the tensorflow object detection API for tensorflow 1.
First of all, you probably would like to use some virtual environment manager. I am using conda here. Asuming you have conda installed (either Anaconda or miniconda) you should proceed as follow:
### Conda environment setup
- Create a conda environment with python 3.7 (it is recommended to use python3.7 to reduce compatibility problems with tensorflow 1). It will also install pip 
- unset the pythonpath to avoid conflicts. Conda doesn't play well with pythonpath and it may cause you some headaches if you don't do this
- Activate your environment
- Check that pip is correctly binded to this environment. By listing all pip packages you should see just the minimal ones installed when creating the conda environment. If your list includes other packages, pip is not correctly binded to this conda environment.
```sh
# create the environment with python 3.7
$ conda create -n tf1 python=3.7
# unset the PYTHONPATH to avoid confilcts and activate the environment
$ unset PYTHONPATH
$ conda activate tf1
# list pip packages to see if pip is correctly binded to our conda environment
$ pip list
```
As a sanity check I would suggest to launch a python shell and typing
```sh
>>>import sys
>>>print(sys.executable)
```
This sould print the path to your anaconda environment and not the path to another python installation you have somewhere else.
To find where python, python3, pip and pip3 are pointing you can try something like this in your bash
```sh
$ for i in pip pip3 python python3 ; do type $i ; done
```
### Installing needed libraries
We will need to manually install some libraries. Some others will be automagically installed when installing the tensorflow object detection API in the next section. The libraries we are manually installing are:
| library | purpose |
| ------ | ------ |
| labelme |  Using labelme utilities
| ipykernel | Adding conda environment as a jupyter kernel (it also installs jupyter) |
| tensorflow 1.15 | Well, you know... we are using tensorflow's object detection API |
| tensorflow-gpu 1.15 | In case you have a suitable GPU |

```sh
$ pip install labelme ipykernel tensorflow==1.15
```
or
```sh
$ pip install labelme ipykernel tensorflow-gpu==1.15
```


### Adding the conda environment as a jupyter kernel
Now that we have instaled juypter and ipykernel we can install our conda envirnment as a jupyter kernel so that the notbook which we will use can work properly.
```sh
$ python3 -m ipykernel install --user --name=tf1
```
I am using python3 insetad of python because, by running the command at the end of the previous section "Conda environment setup", I saw that python3 and not python is binded to the conda environment.

### Installing tensorflow object detection API and all its dependecies
We need to install the object detection API which is just a pack of scripts with utilities to train object detection models using tensorflow. In order to install the API I followed their guide on https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md but I am reviwing it here to centralise the information.
Within this project folder clone tensorflow repo. You don't need the whole repo, only the object detection folder, but they recommend you to install the whole repository in their guide and thats what we will do.
```sh
$ git clone https://github.com/marloquemegusta/tensorflow_object_detection_guide.git
```
Our directory structure now should look like this:
![project structure](https://github.com/marloquemegusta/tensorflow_object_detection_guide/blob/master/project_structure_with_models_folder.PNG?raw=true)
Now move to models/research  and compile the protos. This will create one .py file out of each .pb file in order to use them as python importable modules
```sh
# move to research folder
$ cd models/research
# Compile protos.
$ protoc object_detection/protos/*.proto --python_out=.
```

The final step is to install the TensorFlow Object Detection API y running
```sh
$ pip install --use-feature=2020-resolver .
```
And thest it
```sh
$ python3 object_detection/builders/model_builder_tf1_test.py
```
