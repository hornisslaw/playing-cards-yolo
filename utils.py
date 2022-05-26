import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import CARD_CLASSES, ANCHORS, CONF_THRESHOLD, DEVICE


def iou_width_height(boxes1, boxes2):
    # boxes1 (tensor): width and height of the first bounding boxes
    # boxes2 (tensor): width and height of the second bounding boxes
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    intersection_over_union = intersection / union

    return intersection_over_union


def intersection_over_union(boxes_preds, boxes_labels, epsilon=1e-6):
    # boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
    # boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    # box_format: (x,y,w,h)

    # For Yolo format we have to transform it to corners
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp 0 is for the case when they do not intersect
    # then intersetction will be 0

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection + epsilon
    intersection_over_union = intersection / union

    return intersection_over_union

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = CARD_CLASSES

    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert (
            len(box) == 6
        ), "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()

def non_max_suppression(bboxes, iou_threshold=1, threshold=0.7, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(
        BATCH_SIZE, num_anchors * S * S, 6
    )
    return converted_bboxes.tolist()


def get_evaluation_bboxes(predictions, labels, batch_size, device=DEVICE):
    preds = {
        "boxes" : [],
        "scores" : [],
        "labels" : []}

    targets = {
        "boxes" : [],
        "labels" : []}
    bboxes = [[] for _ in range(batch_size)]
 
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*ANCHORS[i]]).to(device) * S
        boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)
 
    for idx in range(batch_size):
        nms_boxes = non_max_suppression(bboxes[idx])

        # box is specified as [class_prediction, prob_score, x1, y1, x2, y2]
        for nms_box in nms_boxes:
            preds["boxes"].append(nms_box[2:])
            preds["labels"].append(nms_box[0])
            preds["scores"].append(nms_box[1])
 
        for box in true_bboxes[idx]:
            if box[1] > CONF_THRESHOLD:
                targets["boxes"].append(box[2:])
                targets["labels"].append(box[0])
    
    for k, v in preds.items():
        preds[k] = torch.tensor(v, device=DEVICE)
    
    for k, v in targets.items():
        targets[k] = torch.tensor(v, device=DEVICE)
    
    return preds, targets


# def get_evaluation_bboxes(predictions, labels, batch_size, device=DEVICE):
#     train_idx = 0
#     all_pred_boxes = []
#     all_true_boxes = []
#     bboxes = [[] for _ in range(batch_size)]
#     for i in range(3):
#         S = predictions[i].shape[2]
#         anchor = torch.tensor([*ANCHORS[i]]).to(device) * S
#         boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
#         for idx, (box) in enumerate(boxes_scale_i):
#             bboxes[idx] += box

#     # we just want one bbox for each label, not one for each scale
#     true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

#     for idx in range(batch_size):
#         nms_boxes = non_max_suppression(bboxes[idx])

#         for nms_box in nms_boxes:
#             all_pred_boxes.append([train_idx] + nms_box)

#         for box in true_bboxes[idx]:
#             if box[1] > CONF_THRESHOLD:
#                 all_true_boxes.append([train_idx] + box)

#             train_idx += 1

#     return all_pred_boxes, all_true_boxes

