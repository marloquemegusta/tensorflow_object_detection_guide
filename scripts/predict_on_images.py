import numpy as np
import os
import tensorflow as tf
from distutils.version import StrictVersion
import glob
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
import sys
from PIL import Image
    

if StrictVersion(tf.__version__) != StrictVersion('1.14.0'):
    raise ImportError('Please install tensorflow 1.14.0')

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_path", required=True,
                help="path to the input model")
ap.add_argument("-o", "--output_directory", required=True,
                help="path to the output directory")
ap.add_argument("-l", "--path_to_labels", required=True,
                help="path to the labels.pbtxt file")
ap.add_argument("-i", "--input_directory", required=True,
                help="path to the input directory")
ap.add_argument("--skip_boxes", action="store_true",
                help="flag to indicate that bounding boxes shouldnt be visualized")
ap.add_argument("--skip_labels", action="store_true",
                help="flag to indicate that labels shouldnt be visualized")
ap.add_argument("--skip_masks", action="store_true",
                help="flag to indicate that masks shouldnt be visualized")
ap.add_argument("--skip_scores", action="store_true",
                help="flag to indicate that scores shouldnt be visualized")
args = vars(ap.parse_args())

if not glob.glob(args["output_directory"]):
    print("output directory does not exist, creating it")
    os.makedir(args["output_directory"])
if not glob.glob(args["input_directory"]):
    sys.exit("output dir does not exist")

PATH_TO_FROZEN_GRAPH = args["model_path"] + '/frozen_inference_graph.pb'

# Load a (frozen) Tensor    flow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print("succesfully opened frozen graph")
# loading label map
category_index = label_map_util.create_category_index_from_labelmap(args["path_to_labels"], use_display_name=True)
print("succesfully opened label map")


# detection
def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(tf.cast(detection_masks_reframed,tf.float32), 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
# to generate labeled images and store them

# to generate labeled images and store them
try:
    with detection_graph.as_default():
        with tf.Session() as sess:
            for imagepath in glob.glob(args["input_directory"] + "/*.jpg"):
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                imagename = imagepath.split(os.sep)[-1].split(".")[-2]
                image=Image.open(imagepath)
                #image=image.resize((400,round(400/image.width*image.height)))
                image_np = np.asarray(image).copy()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, detection_graph)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    skip_boxes=args["skip_boxes"],
                    skip_scores=args["skip_scores"],
                    skip_labels=args["skip_labels"],
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
    #                 Image.fromarray(image_np).show()
                imagename+= "_boxes" if not args["skip_boxes"] else ""
                imagename+= "_scores" if not args["skip_scores"] else ""
                imagename+= "_labels" if not args["skip_labels"] else ""
                vis_util.save_image_array_as_png(image_np,args["output_directory"] + "/" + imagename + "_mask_predictions.jpg")
                print(args["output_directory"] + "/" + imagename + "_mask_predictions.jpg")
                #break
except Exception as e:
    print(e)