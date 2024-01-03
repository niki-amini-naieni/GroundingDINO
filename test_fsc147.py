from groundingdino.util.inference import load_model, load_image, predict, annotate

import os
import supervision as sv
import json
import numpy as np
from pluralizer import Pluralizer

CONFIG_PATH = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "../weights/groundingdino_swint_ogc.pth"
DATA_SPLIT_PATH = "../Train_Test_Val_FSC_147.json"
IMG_DIR = "../images_384_VarV2"
CLASS_NAME_PATH = "../ImageClasses_FSC147.txt"
FSC147_ANNO_FILE = "../annotation_FSC147_384.json"
FSC147_D_ANNO_FILE = "../CounTX-plusplus/FSC-147-D.json"
DATA_SPLIT = "val"
descriptions = "fsc147d"

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
      class_dict[img_name] = pluralizer.singular(fsc147_d_annotations[img_name]["text_description"])

model = load_model(CONFIG_PATH, WEIGHTS_PATH)
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.35

abs_errs = []
sq_errs = []

for img_name in image_names:
  image_source, image = load_image(IMG_DIR + "/" + img_name)
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

  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  print(type(annotated_frame))
