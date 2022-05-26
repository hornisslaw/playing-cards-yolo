import torch
from config import CONF_THRESHOLD, DEVICE


def check_class_accuracy(outputs, labels, epsilon=1e-16, conf_threshold=CONF_THRESHOLD):
    total_class_preds, total_noobj, total_obj = 0, 0, 0
    correct_class, correct_noobj, correct_obj = 0, 0, 0

    for out, label in zip(outputs, labels):
        label = label.to(DEVICE)
        is_object = label[..., 0] == 1
        no_object = label[..., 0] == 0

        correct_class += torch.sum(
            torch.argmax(out[..., 5:][is_object], dim=-1) == label[..., 5][is_object]
        )
        obj_preds = torch.sigmoid(out[..., 0]) > conf_threshold
        correct_obj += torch.sum(obj_preds[is_object] == label[..., 0][is_object])
        correct_noobj += torch.sum(obj_preds[no_object] == label[..., 0][no_object])

        total_class_preds += torch.sum(is_object)
        total_obj += torch.sum(is_object)
        total_noobj += torch.sum(no_object)

    class_accuracy = correct_class / (total_class_preds + epsilon)
    no_obj_accuracy = correct_noobj / (total_noobj + epsilon)
    obj_accuracy = correct_obj / (total_obj + epsilon)

    return (class_accuracy, no_obj_accuracy, obj_accuracy)
