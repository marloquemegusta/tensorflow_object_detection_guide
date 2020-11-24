import cv2
import numpy as np


def predict_on_single_image(image, model, masks=False, size=(600, 600)):
    model.setInput(cv2.dnn.blobFromImage(image, size=size, swapRB=True, crop=False))
    if masks:
        return model.forward(["detection_out_final", "detection_masks"])
    else:
        return model.forward()


def pick_first_traviesa(boxes, limit, threshold):
    cond1 = boxes[0, 0, :, 2] > threshold
    cond2 = boxes[0, 0, :, 6] < limit
    valid_boxes = boxes[0, 0][np.logical_and(cond1, cond2)]
    traviesas = valid_boxes[valid_boxes[:, 1] == 0]
    # si no encontramos traviesa devolvemos None
    if traviesas.shape[0] == 0:
        return None
    primera_traviesa = traviesas[traviesas[:, 6] == np.max(traviesas[:, 6])][0]
    cond3 = valid_boxes[:, 6] > primera_traviesa[4]
    cond4 = valid_boxes[:, 4] < primera_traviesa[6]
    valid_boxes = valid_boxes[np.logical_and(cond3, cond4)]
    return valid_boxes.reshape(1, 1, valid_boxes.shape[0], valid_boxes.shape[1])


def visualize_predictions(image, colors, labels, boxes, masks=None, exclude_masks=[], exclude_boxes=[], threshold=0.5):
    W = image.shape[1]
    H = image.shape[0]
    clone = image.copy()
    class_mask = np.zeros(image.shape).astype("uint8")
    boxes_to_draw = []
    for i in range(boxes.shape[2]):
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        if confidence > threshold:
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            color = colors[classID]
            if (classID not in exclude_masks) and (masks is not None):
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH),
                                  interpolation=cv2.INTER_CUBIC)
                mask = (mask > threshold)
                class_mask[startY:endY, startX:endX][mask] = color
                # store the blended ROI in the original image
            if classID not in exclude_boxes:
                boxes_to_draw.append(((startX, startY), (endX, endY), classID))
    clone[class_mask != (0, 0, 0)] = clone[class_mask != (0, 0, 0)] * 0.4 + class_mask[class_mask != (0, 0, 0)] * 0.6
    for box_to_draw in boxes_to_draw:
        color = colors[box_to_draw[2]]
        color = color.tolist()[0], color.tolist()[1], color.tolist()[2]
        cv2.rectangle(clone, box_to_draw[0], box_to_draw[1], color, 3)
        text = labels[box_to_draw[2]]
        cv2.putText(clone, text, (box_to_draw[0][0], box_to_draw[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return clone


def crop_traviesa(image, boxes):
    boxes = boxes.copy()
    H = image.shape[0]
    W = image.shape[1]
    boxes = boxes[0, 0, 0]
    boxes[3], boxes[4], boxes[5], boxes[6] = np.multiply((boxes[3], boxes[4], boxes[5], boxes[6]), (W, H, W, H))
    boxes = np.array(boxes).astype("int")
    return image[boxes[4]:boxes[6], boxes[3]:boxes[5]]


def generate_incidence(boxes, W, H, labels):
    #id_to_label = {i:labels[i] for i in range(len(labels))}
    recuento_traviesas = 0
    recuento_clips = 0
    recuento_tornillos = 0
    data={}
    boxes=boxes.copy()[0][0]
    data["detections"] = list(range(0, len(boxes)))
    for i in range(boxes.shape[0]):
        label = labels[int(boxes[i, 1])]
        boxes[i][3:7] = boxes[i][3:7] * np.array([W, H, W, H])
        upper_left = (float(boxes[i, 3]), float(boxes[i, 4]))
        bottom_right = (float(boxes[i, 5]), float(boxes[i, 6]))
        data["detections"][i] = {"label": label, "points": [upper_left, bottom_right]}
        recuento_traviesas = recuento_traviesas + 1 if label == "traviesa" else recuento_traviesas
        recuento_clips = recuento_clips + 1 if label == "clip" else recuento_clips
        recuento_tornillos = recuento_tornillos + 1 if label == "tornillo" else recuento_tornillos
        data["incidencias"] = {"traviesa": 1 - recuento_traviesas, "clip": 2 - recuento_clips,
                               "tornillos": 2 - recuento_tornillos}
    return data