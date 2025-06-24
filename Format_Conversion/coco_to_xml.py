import argparse
import os
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import numpy as np
 
parser = argparse.ArgumentParser(description='COCO JSON to Pascal VOC XML Converter (with segmentation support).')
parser.add_argument('--coco_json', required=True, help='COCO json file (with detection/segmentation annotations).')
parser.add_argument('--coco_folder', required=False, default='', help='Folder containing the images.')
parser.add_argument('--save_xml', required=True, help='Folder to save Pascal VOC XML files.')
parser.add_argument('--database_name', required=False, default='', help='Name of dataset.')
parser.add_argument('--no_skip_background', dest='skip_background', action='store_false',
                    help='Do not skip background category (if present).')
parser.set_defaults(skip_background=True)
 
args = parser.parse_args()
 
def rle_to_polygon(rle):
    mask = mask_utils.decode(rle)
    contours = []
    from skimage import measure
    for contour in measure.find_contours(mask, 0.5):
        contour = np.flip(contour, axis=1)
        points = contour.ravel().astype(int)
        if len(points) >= 6:  # Minimum 3 points
            contours.append(points)
    return contours
 
def write_to_xml(image_name, bboxes, image_folder_name, data_folder, save_folder, database_name):
    with Image.open(os.path.join(data_folder, image_name)) as img:
        width, height = img.size
        depth = len(img.getbands())
 
    objects = ''
    for bbox in bboxes:
        obj_str = f'''
<object>
<name>{bbox["name"]}</name>
<pose>Unspecified</pose>
<truncated>0</truncated>
<difficult>0</difficult>
<bndbox>
<xmin>{bbox["bbox"][0]}</xmin>
<ymin>{bbox["bbox"][1]}</ymin>
<xmax>{bbox["bbox"][2]}</xmax>
<ymax>{bbox["bbox"][3]}</ymax>
</bndbox>'''
 
        if "segmentation" in bbox:
            for poly in bbox["segmentation"]:
                polygon_str = ' '.join(str(x) for x in poly.tolist())
                obj_str += f'''
        <segmentation>
            <polygon>{polygon_str}</polygon>
        </segmentation>'''
 
        obj_str += '\n\t</object>'
        objects += obj_str
 
    xml = f'''<annotation>
<folder>{image_folder_name}</folder>
<filename>{image_name}</filename>
<source>
<database>{database_name}</database>
</source>
<size>
<width>{width}</width>
<height>{height}</height>
<depth>{depth}</depth>
</size>
<segmented>1</segmented>{objects}
</annotation>'''
 
    anno_path = os.path.join(save_folder, os.path.splitext(image_name)[0] + '.xml')
    with open(anno_path, 'w') as file:
        file.write(xml)
 
# Setup
Path(args.save_xml).mkdir(parents=True, exist_ok=True)
image_folder_name = os.path.basename(os.path.abspath(args.coco_folder))
coco = COCO(args.coco_json)
imgIds = coco.getImgIds()
 
if not args.database_name:
    args.database_name = image_folder_name
 
print('Writing annotation files...')
 
for idx, imgId in enumerate(imgIds, 1):
    img = coco.loadImgs(imgId)[0]
    img_path = os.path.join(args.coco_folder, img['file_name'])
 
    if not os.path.exists(img_path):
        print(f"Image {img['file_name']} not found in folder.")
        continue
 
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    bboxes = []
 
    for ann in anns:
        cat = coco.loadCats(ann['category_id'])[0]
        if cat['name'] == 'background' and args.skip_background:
            continue
 
        x, y, w, h = ann['bbox']
        bbox_info = {
            "name": cat['name'],
            "bbox": [int(x), int(y), int(x + w), int(y + h)]
        }
 
        if "segmentation" in ann and isinstance(ann["segmentation"], dict):  # RLE format
            rle = ann["segmentation"]
            polys = rle_to_polygon(rle)
            if polys:
                bbox_info["segmentation"] = polys
 
        bboxes.append(bbox_info)
 
    write_to_xml(img['file_name'], bboxes, image_folder_name, args.coco_folder, args.save_xml, args.database_name)
    print(f"({idx}/{len(imgIds)}) Wrote: {img['file_name']}")
 
print("Pascal VOC XML annotations saved.")
