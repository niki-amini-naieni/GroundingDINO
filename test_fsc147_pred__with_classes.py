from groundingdino.util.inference import load_model, load_image, predict, annotate, Model

import os
import supervision as sv
import json
import numpy as np
from pluralizer import Pluralizer
import cv2

CONFIG_PATH = "./groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "../weights/groundingdino_swinb_cogcoor.pth"
DATA_SPLIT_PATH = "/scratch/shared/beegfs/nikian/FSC-147/Train_Test_Val_FSC_147.json"
IMG_DIR = "/scratch/shared/beegfs/nikian/FSC-147/images_384_VarV2"
CLASS_NAME_PATH = "/scratch/shared/beegfs/nikian/FSC-147/ImageClasses_FSC147.txt"
FSC147_ANNO_FILE = "/scratch/shared/beegfs/nikian/FSC-147/annotation_FSC147_384.json"
FSC147_D_ANNO_FILE = "../CounTX-plusplus/FSC-147-D.json"
DATA_SPLIT = "test"
descriptions = "fsc147"

with open(DATA_SPLIT_PATH) as f:
    data_split = json.load(f)
image_names = data_split[DATA_SPLIT]

with open(FSC147_D_ANNO_FILE) as f:
    fsc147_d_annotations = json.load(f)

with open(FSC147_ANNO_FILE) as f:
    fsc147_annotations = json.load(f)

pluralizer = Pluralizer()

class_dict = {}

if descriptions == "fsc147":
  with open(CLASS_NAME_PATH) as f:
      for line in f:
          key = line.split()[0]
          val = line.split()[1:]
          class_dict[key] = pluralizer.singular(' '.join(val))
          #class_dict[key] = ' '.join(val)
else:
   for img_name in image_names:
      class_dict[img_name] = pluralizer.singular(fsc147_d_annotations[img_name]["text_description"][4:])

classes = list(np.unique(list(class_dict.values())))

BOX_THRESHOLD = 0.1
TEXT_THRESHOLD = 0.25
model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)

abs_errs = []
sq_errs = []
im_sink = sv.utils.image.ImageSink(target_dir_path="/users/nikian/GroundingDINO")
for img_name in image_names:
  image = cv2.imread(IMG_DIR + "/" + img_name)
  gt = len(fsc147_annotations[img_name]["points"])
  detections = model.predict_with_classes(
            image=image,
            classes=[class_dict[img_name]],
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
  pred = len(list(filter(lambda x: x is not None and (x == 0), detections.class_id)))
  pred = len(detections.class_id)

  abs_err = np.abs(pred - gt)
  abs_errs.append(abs_err)
  sq_errs.append(abs_err ** 2)
  if img_name == "5.jpg" or img_name == "287.jpg":
    print("Pred: " + str(pred))
    print("GT: " + str(gt))
    print("Abs Err: " + str(abs_err))
    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    im_sink.save_image(annotated_image, img_name)

abs_errs = np.array(abs_errs)
sq_errs = np.array(sq_errs)
print(abs_errs)
print(sq_errs)
print("MAE: " + str(np.mean(abs_errs)))
print("RMSE: " + str(np.sqrt(np.mean(sq_errs))))
