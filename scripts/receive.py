import stomp
import json
import time
import cv2
import utilities
import numpy as np

model_path_traviesas = "models/inference_graph_faster_rcnn_24_11"
model_path_soldaduras = "models/inference_graph_faster_rcnn_18_11"
cvNet_traviesas = cv2.dnn.readNetFromTensorflow(model_path_traviesas + '/frozen_inference_graph.pb',
                                          model_path_traviesas + '/graph.pbtxt')
cvNet_soldaduras = cv2.dnn.readNetFromTensorflow(model_path_soldaduras + '/frozen_inference_graph.pb',
                                          model_path_soldaduras + '/graph.pbtxt')

threshold=0.9
labels_traviesa = ["traviesa","clip","tornillo","marca"]
labels_soldadura = ["soldadura_1", "soldadura_2"]
num_classes = max(len(labels_traviesa), len(labels_soldadura)) # OJO SOLO PARA COLORES
available_colors = np.random.randint(0, 255, (num_classes, 3))
output_dir = "prueba/"

class MyListener_traviesa(stomp.ConnectionListener):
    def on_error(self, headers, message):
        print('received an error "%s"' % message)
    def on_message(self, headers, message, labels=labels_traviesa):
        data = json.loads(str(message))
        print("mensaje de {} recibido".format(data["camara"]))
        img = cv2.imread(data['path'])
        if img is None:
            print("image path with id {} recieved from {} is not valid".format(data["idCaptura"], data["camara"]))
            return None

        detections = utilities.predict_on_single_image(img, cvNet_traviesas)
        first_traviesa_detections = utilities.pick_largest_traviesa(detections, threshold)
        if first_traviesa_detections is not None:
            image_with_detections = utilities.visualize_predictions(image=img,
                                                                    colors=available_colors,
                                                                    labels=labels,
                                                                    boxes=first_traviesa_detections,
                                                                    threshold=threshold)
            incidence_data = utilities.generate_incidence(boxes=first_traviesa_detections,
                                                          W=img.shape[1],
                                                          H=img.shape[0],
                                                          labels=labels)

            incidence_data['path_input_image'] = data['path']
            incidence_data['path_output_image'] = output_dir + data["idCaptura"] + "_detections.JPG"
            incidence_data['detection_type'] = "object_detection"

            cv2.imwrite(output_dir + data["idCaptura"] + "_detections.JPG", image_with_detections)
            with open(output_dir + data["idCaptura"] + "_results.json",
                      'w') as outfile:
                json.dump(incidence_data, outfile, indent=4)
            print("succesfullly analized {} img from {}".format(data["idCaptura"], data["camara"]))


hosts = [('localhost', 61613)]
conn_traviesa = stomp.Connection(host_and_ports=hosts)
conn_traviesa.set_listener('', MyListener_traviesa())
conn_traviesa.start()
conn_traviesa.connect(wait=True)

conn_traviesa.subscribe(destination='/queue/test', id=1, ack='auto')

time.sleep(2)
conn_traviesa.disconnect()