import json
from pathlib import Path
import conllu
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gold", action="store_true", help="whether to use gold boxes instead of proposals"
)
parser.add_argument(
    "--file",
    help="path to the prediction",
    default="outputs/0_latest_run/dev.predict.txt",
)
parser.add_argument(
    "--dataroot",
    help="path to VLParse",
    default="data/vlparse",
)
args = parser.parse_args()

id_list_path = f"{args.dataroot}/id_list/val.txt"
predict_path = args.file

if args.gold:
    with open(f"{args.dataroot}/dev_gold_boxes.json") as f:
        img2boxes = json.load(f)
else:
    with open(f"{args.dataroot}/dev_roi_boxes.json") as f:
        img2boxes = json.load(f)
img2boxes = {int(key): value for key, value in img2boxes.items()}

with open(f"{args.dataroot}/vlparse.json") as f:
    gold = json.load(f)
gold = {item["coco_id"]: item for item in gold if isinstance(item, dict)}


id_list = [line for line in Path(id_list_path).read_text().splitlines()]
img_ids = [int(item) for item in id_list for _ in range(5)]
sent_ids = [item for _ in id_list for item in range(5)]
predict = list(
    conllu.parse_incr(open(predict_path), fields=["ID", "FORM", "POS", "HEAD", "ALIGN"])
)
has_vg = [item in gold for item in img_ids]
img_ids = [item for item, flag in zip(img_ids, has_vg) if flag]
sent_ids = [item for item, flag in zip(sent_ids, has_vg) if flag]
# predict = [item for item, flag in zip(predict, has_vg) if flag]
print(len(sent_ids), len(predict))


def get_position(item):
    return item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"]


def bb_intersection_over_union(boxA, boxB):
    # boxA = [int(x) for x in boxA]
    # boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


obj_correct = 0
obj_total = 0
attr_correct = 0
attr_total = 0
rel_correct = 0
rel_total = 0


def test(boxA, boxB):
    if bb_intersection_over_union(boxA, boxB) >= 0.5:
        return True
    return False


for idx in range(len(predict)):
    img_id, sent_id = img_ids[idx], sent_ids[idx]

    # obj
    for word_idx, data in gold[img_id]["txt2sg"][sent_id].items():
        if data["type"] != "OBJ":
            continue
        correct_flag = False
        for item in predict[idx][int(word_idx)]["ALIGN"].split("|"):
            pred_type, pred_id = item.split()
            if pred_type == "obj":
                word_predict = img2boxes[img_id][int(pred_id)]
                correct_flag = False
                for obj_id, _ in data["candidates"]:
                    position = get_position(gold[img_id]["obj"][obj_id])
                    if test(word_predict, position):
                        correct_flag = True
                        break
                if correct_flag:
                    obj_correct += 1
                    break
        obj_total += 1

    # attr
    for word_idx, data in gold[img_id]["txt2sg"][sent_id].items():
        if data["type"] != "ATTR":
            continue
        correct_flag = False
        for item in predict[idx][int(word_idx)]["ALIGN"].split("|"):
            pred_type, pred_id = item.split()
            if pred_type == "attr":
                try:
                    word_predict = img2boxes[img_id][int(pred_id)]
                except IndexError:
                    print(img_id, sent_id)
                correct_flag = False
                for obj_id, _ in data["candidates"]:
                    position = get_position(gold[img_id]["obj"][obj_id])
                    if test(word_predict, position):
                        correct_flag = True
                        break
                if correct_flag:
                    attr_correct += 1
                    break
        attr_total += 1

    # rel
    for word_idx, data in gold[img_id]["txt2sg"][sent_id].items():
        if data["type"] != "REL":
            continue
        correct_flag = False
        for item in predict[idx][int(word_idx)]["ALIGN"].split("|"):
            pred_type, pred_id = item.split()
            if pred_type == "rel":
                obj1, obj2 = pred_id.split("-")
                obj1 = img2boxes[img_id][int(obj1)]
                obj2 = img2boxes[img_id][int(obj2)]

                correct_flag = False
                for rel_id, _ in data["candidates"]:
                    rel_item = gold[img_id]["rel"][rel_id - len(gold[img_id]["obj"])]
                    assert rel_item["id"] == rel_id
                    gold_obj1 = get_position(gold[img_id]["obj"][rel_item["subj"]])
                    gold_obj2 = get_position(gold[img_id]["obj"][rel_item["obj"]])

                    if test(obj1, gold_obj1) and test(obj2, gold_obj2):
                        correct_flag = True
                        break
                    if test(obj2, gold_obj1) and test(obj1, gold_obj2):
                        correct_flag = True
                        break
                if correct_flag:
                    rel_correct += 1
                    break
        rel_total += 1


print("obj", obj_correct / obj_total, obj_total)
print("attr", attr_correct / attr_total, attr_total)
print("rel", rel_correct / rel_total, rel_total)
print(
    "0-order",
    (obj_correct + attr_correct + rel_correct) / (obj_total + attr_total + rel_total),
)
