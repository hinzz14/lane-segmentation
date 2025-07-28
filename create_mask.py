import os
import cv2
import numpy as np
from pycocotools.coco import COCO

image_dir = 'dataset/train'

annotation_file = 'dataset/train/_annotations.coco.json'

output_dir = 'dataset/train_mask'

os.makedirs(output_dir, exist_ok=True)

coco = COCO(annotation_file)

img_ids = coco.getImgIds()
images = coco.loadImgs(img_ids)

for i, img_info in enumerate(images):
    img_id = img_info['id']
    img_file_name = img_info['file_name']
    img_width = img_info['width']
    img_height = img_info['height']

    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        category_name = coco.loadCats(ann['category_id'])[0]['name']

        color_value = 0
        if 'solid' in category_name.lower():  
            color_value = 1                 
        elif 'dashed' in category_name.lower(): 
            color_value = 2                 
        else:
            continue

        segmentation = ann['segmentation']
        
        polygons = [np.array(poly).reshape((-1, 2)).astype(np.int32) for poly in segmentation]

        cv2.fillPoly(mask, polygons, color_value)

    output_path = os.path.join(output_dir, img_file_name.replace('.jpg', '.png')) # Lưu dưới dạng PNG
    cv2.imwrite(output_path, mask)
