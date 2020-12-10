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


def pick_largest_traviesa(boxes, threshold):
    valid_boxes = boxes[0, 0][boxes[0, 0, :, 2] > threshold]
    traviesas = valid_boxes[valid_boxes[:, 1] == 0]
    if traviesas.shape[0] == 0:
        return None
    max_area = 0
    largest_traviesa = np.zeros(7)
    for traviesa in traviesas:
        area = np.abs(traviesa[3] - traviesa[5]) * np.abs(traviesa[4] - traviesa[6])
        if area > max_area:
            max_area = area
            largest_traviesa = traviesa
    # si no encontramos traviesa devolvemos None
    cond1 = valid_boxes[:, 6] > largest_traviesa[4]
    cond2 = valid_boxes[:, 4] < largest_traviesa[6]
    valid_boxes = valid_boxes[np.logical_and(cond1, cond2)]
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
        cv2.rectangle(clone, box_to_draw[0], box_to_draw[1], color, 6)
        text = labels[box_to_draw[2]]
        cv2.putText(clone, text, (box_to_draw[0][0], box_to_draw[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    return clone


def crop_element(image, boxes, label):
    if boxes is None:
        return None
    boxes = boxes.copy()
    boxes = boxes[0, 0][boxes[0, 0, :, 1] == label]
    boxes = boxes[0]
    H = image.shape[0]
    W = image.shape[1]
    boxes[3], boxes[4], boxes[5], boxes[6] = np.multiply((boxes[3], boxes[4], boxes[5], boxes[6]), (W, H, W, H))
    boxes = np.array(boxes).astype("int")
    return image[boxes[4]:boxes[6], boxes[3]:boxes[5]]


def crop_custom(image, boxes):
    if boxes is None:
        return None
    boxes = boxes.copy()
    H = image.shape[0]
    W = image.shape[1]
    clips = boxes[0, 0][boxes[0, 0, :, 1] == 1]
    trav = boxes[0, 0][boxes[0, 0, :, 1] == 0][0]
    max_clip_height = 0
    black_coords = [99999, 99999, 0, 0]
    for clip in clips:
        if np.abs(clip[4] - clip[6]) > max_clip_height:
            max_clip_height = np.abs(clip[4] - clip[6])
            largest_clip_coords = clip[3:7]
        black_coords[0] = clip[3] if clip[3] < black_coords[0] else black_coords[0]
        black_coords[3] = clip[6] if clip[6] > black_coords[3] else black_coords[3]
        black_coords[1] = clip[4] if clip[4] < black_coords[1] else black_coords[1]
        black_coords[2] = clip[5] if clip[5] > black_coords[2] else black_coords[2]
    black_coords = np.multiply(black_coords, [W, H, W, H]).astype("int")
    largest_clip_coords = np.multiply(largest_clip_coords, [W, H, W, H]).astype("int")
    trav[3], trav[5] = np.multiply((trav[3], trav[5]), (W, W))
    startX, endX = trav[3].astype("int"), trav[5].astype("int")
    startY, endY = largest_clip_coords[1], largest_clip_coords[3]
    image[black_coords[1]:black_coords[3], black_coords[0]:black_coords[2]] = [0, 0, 0] if image.shape[2] == 3 else 0
    return image[startY:endY, startX:endX]


def crop_custom_marca(image, boxes, return_mask=0):
    if boxes is None:
        return None
    boxes = boxes.copy()
    H = image.shape[0]
    W = image.shape[1]
    max_clip_height = 0
    clips = boxes[0, 0][boxes[0, 0, :, 1] == 1]
    marca = boxes[0, 0][boxes[0, 0, :, 1] == 3]
    trav = boxes[0, 0][boxes[0, 0, :, 1] == 0][0]
    black_coords = [99999, 99999, 0, 0]

    for clip in clips:
        if np.abs(clip[4] - clip[6]) > max_clip_height:
            max_clip_height = np.abs(clip[4] - clip[6])
            largest_clip_coords = clip[3:7]
        black_coords[0] = clip[3] if clip[3] < black_coords[0] else black_coords[0]
        black_coords[3] = clip[6] if clip[6] > black_coords[3] else black_coords[3]
        black_coords[1] = clip[4] if clip[4] < black_coords[1] else black_coords[1]
        black_coords[2] = clip[5] if clip[5] > black_coords[2] else black_coords[2]

    largest_clip_coords = np.multiply(largest_clip_coords, [W, H, W, H]).astype("int")
    black_coords = np.multiply(black_coords, [W, H, W, H]).astype("int")

    trav[3], trav[5] = np.multiply((trav[3], trav[5]), (W, W))
    trav[4], trav[6] = np.multiply((trav[4], trav[6]), (H, H))
    startX, endX = trav[3], trav[5]

    if len(marca) == 0:
        startY, endY = largest_clip_coords[1], largest_clip_coords[3]
    else:
        marca = marca[0]
        marca[4], marca[6] = np.multiply((marca[4], marca[6]), (H, H))
        if (np.abs(marca[4] - marca[6]) > np.abs(black_coords[1] - black_coords[3]) * 0.8):
            startY, endY = marca[4].astype("int"), marca[6].astype("int")
        else:
            startY, endY = (marca[4] * 1.05).astype("int"), (marca[6] * 1.05).astype("int")

    if return_mask:
        mask = np.ones((H, W))
        mask[0:startY, 0:W] = 0
        mask[endY:H, 0:W] = 0
        mask[startY:endY, black_coords[0]:black_coords[2]] = 0
        image = image[trav[4].astype("int"):trav[6].astype("int"), trav[3].astype("int"):trav[5].astype("int")]
        mask = mask[trav[4].astype("int"):trav[6].astype("int"), trav[3].astype("int"):trav[5].astype("int")]
        return image, mask
    else:
        # put a black square instead of the clips
        image[startY:endY, black_coords[0]:black_coords[2]] = [0, 0, 0] if image.shape[2] == 3 else 0
        return image[startY:endY, startX:endX]


def generate_incidence(boxes, W, H, labels):
    # id_to_label = {i:labels[i] for i in range(len(labels))}
    recuento_traviesas = 0
    recuento_clips = 0
    recuento_tornillos = 0
    data = {}
    boxes = boxes.copy()[0][0]
    data["detections"] = list(range(0, len(boxes)))
    for i in range(boxes.shape[0]):
        label = labels[int(boxes[i, 1])]
        boxes[i][3:7] = boxes[i][3:7] * np.array([W, H, W, H])
        upper_left = (float(boxes[i, 3]), float(boxes[i, 4]))
        bottom_right = (float(boxes[i, 5]), float(boxes[i, 6]))
        data["detections"][i] = {"nombreObjecto": label, "coordenadas": [upper_left, bottom_right],
                                 "porcentaje" : boxes[i][2]}
        recuento_traviesas = recuento_traviesas + 1 if label == "traviesa" else recuento_traviesas
        recuento_clips = recuento_clips + 1 if label == "clip" else recuento_clips
        recuento_tornillos = recuento_tornillos + 1 if label == "tornillo" else recuento_tornillos
        data["incidencias"] = {"traviesa": 1 - recuento_traviesas, "clip": 2 - recuento_clips,
                               "tornillos": 2 - recuento_tornillos}
    return data
