from groundingdino.util.inference import load_model, load_image, predict, annotate

import os
import supervision as sv
import json
import numpy as np
from pluralizer import Pluralizer

CONFIG_PATH = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "../weights/groundingdino_swint_ogc.pth"
DATA_SPLIT_PATH = "/scratch/shared/beegfs/nikian/FSC-147/Train_Test_Val_FSC_147.json"
IMG_DIR = "/scratch/shared/beegfs/nikian/FSC-147/images_384_VarV2"
CLASS_NAME_PATH = "/scratch/shared/beegfs/nikian/FSC-147/ImageClasses_FSC147.txt"
FSC147_ANNO_FILE = "/scratch/shared/beegfs/nikian/FSC-147/annotation_FSC147_384.json"
FSC147_D_ANNO_FILE = "../CounTX-plusplus/FSC-147-D.json"
DATA_SPLIT = "val"
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
          #class_dict[key] = pluralizer.singular(' '.join(val))
          class_dict[key] = ' '.join(val)
else:
   for img_name in image_names:
      class_dict[img_name] = pluralizer.singular(fsc147_d_annotations[img_name]["text_description"][4:])

model = load_model(CONFIG_PATH, WEIGHTS_PATH)
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.35

abs_errs = []
sq_errs = []
im_sink = sv.utils.image.ImageSink(target_dir_path="/users/nikian/GroundingDINO")
for img_name in image_names:
  image_source, image = load_image(IMG_DIR + "/" + img_name)
  print(image.shape)
  gt = len(fsc147_annotations[img_name]["points"])
  caption = class_dict[img_name] 
  print("Caption: " + caption)
  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=caption,
      box_threshold=BOX_THRESHOLD,
      text_threshold=TEXT_THRESHOLD
  )

  pred = len(boxes)

  print("Pred: " + str(pred))
  print("GT: " + str(gt))
  abs_err = np.abs(pred - gt)
  print("Abs Err: " + str(abs_err))
  abs_errs.append(abs_err)
  sq_errs.append(abs_err ** 2)
  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  #im_sink.save_image(annotated_frame, img_name)

abs_errs = np.array(abs_errs)
sq_errs = np.array(sq_errs)
print(abs_errs)
print(sq_errs)
print("MAE: " + str(np.mean(abs_errs)))
print("RMSE: " + str(np.sqrt(np.mean(sq_errs))))
