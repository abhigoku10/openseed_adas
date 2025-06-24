import json
import os
import argparse
from collections import defaultdict

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]

def convert_segmentation_coco_to_yolo_poly(segmentation, img_width, img_height):
    # segmentation can be list of lists (polygon) or dict (RLE); YOLOv8 only supports polygons.
    if isinstance(segmentation, list):
        # COCO polygons: list of lists
        polys = []
        for poly in segmentation:
            # poly: [x1, y1, x2, y2, ..., xn, yn]
            xy = [(poly[i] / img_width, poly[i+1] / img_height) for i in range(0, len(poly), 2)]
            # flatten
            polys.extend([f"{x:.6f} {y:.6f}" for x, y in xy])
        return polys
    # If RLE, skip (YOLOv8 does not support RLE polygons)
    return []

def main(coco_json_path, output_dir, save_bbox=True, save_seg=True):
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    image_id_to_info = {img['id']: img for img in coco['images']}
    category_ids = [cat['id'] for cat in coco['categories']]
    category_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

    os.makedirs(output_dir, exist_ok=True)
    img_to_annotations = defaultdict(list)
    for ann in coco['annotations']:
        img_to_annotations[ann['image_id']].append(ann)

    for img_id, anns in img_to_annotations.items():
        img_info = image_id_to_info[img_id]
        img_h, img_w = img_info.get('height'), img_info.get('width')
        file_stem = os.path.splitext(img_info['file_name'])[0]
        yolo_lines = []

        for ann in anns:
            cat_id = ann['category_id']
            yolo_cat = category_id_to_yolo[cat_id]

            line = f"{yolo_cat}"

            # Add segmentation polygon (YOLOv8 format)
            if save_seg and 'segmentation' in ann and ann['segmentation']:
                polycoords = convert_segmentation_coco_to_yolo_poly(ann['segmentation'], img_w, img_h)
                if polycoords:
                    line += " " + " ".join(polycoords)
            # Add bounding box (YOLOv5/8 detection format)
            if save_bbox and 'bbox' in ann and ann['bbox']:
                yolo_bbox = convert_bbox_coco_to_yolo(ann['bbox'], img_w, img_h)
                line += " " + " ".join(f"{x:.6f}" for x in yolo_bbox)
            # Only write line if it has more than class id
            if line.strip() != f"{yolo_cat}":
                yolo_lines.append(line.strip())

        # Write to file if any annotations exist
        if yolo_lines:
            with open(os.path.join(output_dir, f"{file_stem}.txt"), 'w') as f:
                f.write('\n'.join(yolo_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert COCO detection/segmentation JSON to YOLO format (.txt per image). Supports polygon segmentation (YOLOv8) and bounding boxes."
    )
    parser.add_argument("json_path", help="Path to COCO-style JSON file")
    parser.add_argument("output_dir", help="Output folder for YOLO .txt files")
    parser.add_argument("--no_bbox", action="store_true", help="Do NOT export bounding boxes (default: export)")
    parser.add_argument("--no_seg", action="store_true", help="Do NOT export segmentation polygons (default: export if present)")
    args = parser.parse_args()

    main(
        args.json_path,
        args.output_dir,
        save_bbox=not args.no_bbox,
        save_seg=not args.no_seg
    )
