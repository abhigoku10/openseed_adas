import os
import json
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BitMasks
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.arguments import load_opt_command
from detectron2.utils.colormap import random_color
from datetime import datetime
from pycocotools import mask as mask_util
 
thing_classes = ["car", "person", "traffic light", "truck", "motorcycle", "bicycle", "sign board", "bus"]
stuff_classes = ['building', 'sky', 'street', 'tree', 'rock', 'sidewalk', 'house', 'mountain', 'grass', 'stone', 'road', 'lane']
 
def setup_model(opt):
    pretrained_pth = os.path.join(opt['WEIGHT'])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    return model
 
def setup_metadata():
    thing_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() for _ in range(len(thing_classes))]
    thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}
    instance_metadata = MetadataCatalog.get("instance").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id
    )
 
    stuff_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x: x for x in range(len(stuff_classes))}
    semantic_metadata = MetadataCatalog.get("semantic").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id
    )
 
    stuff_dataset_id_to_contiguous_id = {x + len(thing_classes): x for x in range(len(stuff_classes))}
    panoptic_metadata = MetadataCatalog.get("panoptic").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id
    )
 
    return {
        "instance": instance_metadata,
        "semantic": semantic_metadata,
        "panoptic": panoptic_metadata
    }
    
def generate_panoptic_annotations(panoptic_seg, segments_info, image_id, segments_folder):
    segments_info_coco = []
    seg_image = Image.fromarray(panoptic_seg.cpu().numpy().astype(np.uint8))
    seg_filename = f"{str(image_id).zfill(12)}.png"
    seg_path = os.path.join(segments_folder, seg_filename)
    os.makedirs(segments_folder, exist_ok=True)
    seg_image.save(seg_path)
 
    for segment in segments_info:
        segment_id = int(segment["id"])
        mask = (panoptic_seg == segment_id).cpu().numpy().astype(np.uint8)
        rle = mask_util.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
 
        area = int(mask.sum())
        y_indices, x_indices = np.where(mask)
        bbox = [int(x_indices.min()), int(y_indices.min()), int(x_indices.max() - x_indices.min()), int(y_indices.max() - y_indices.min())]
 
        #segments_info_coco.append({
            #"id": segment_id,
            #"category_id": int(segment["category_id"]),
            #"area": area,
            #"bbox": bbox,
            #"iscrowd": 0
        #})
 
    annotation = {
        "image_id": image_id,
        "file_name": seg_filename,
        "id": segment_id,
		"category_id": int(segment["category_id"]),
		"area": area,
		"bbox": bbox,
        "segmentation": rle
    }
 
    return annotation
 
def process_image(image_pth, output_root, model, transform, metadata_dict, coco_output, image_id, segment_id):
    image_ori = Image.open(image_pth).convert("RGB")
    width, height = image_ori.size
    image = transform(image_ori)
    image = np.asarray(image)
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()
    batch_inputs = [{'image': images, 'height': height, 'width': width}]
 
    results = {
        "image": os.path.basename(image_pth),
        "objects": {},
        "total_count": 0,
    }
 
    for seg_type, metadata in metadata_dict.items():
        with torch.no_grad():
            demo = None
 
            if seg_type == "instance":
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes, is_eval=True)
                model.model.metadata = metadata
                model.model.sem_seg_head.num_classes = len(thing_classes)
                outputs = model.forward(batch_inputs)
 
                visual = Visualizer(image_ori, metadata=metadata)
                inst_seg = outputs[-1]['instances']
                inst_seg.pred_masks = inst_seg.pred_masks.cpu()
                inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()
                demo = visual.draw_instance_predictions(inst_seg)
 
                seg_output_path = os.path.join(output_root, "InstanceSeg", os.path.basename(image_pth))
 
            elif seg_type == "semantic":
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
                model.model.metadata = metadata
                model.model.sem_seg_head.num_classes = len(stuff_classes)
                outputs = model.forward(batch_inputs, inference_task="sem_seg")
 
                visual = Visualizer(image_ori, metadata=metadata)
                sem_seg = outputs[-1]['sem_seg'].max(0)[1]
                demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5)
 
                seg_output_path = os.path.join(output_root, "SemanticSeg", os.path.basename(image_pth))
 
            elif seg_type == "panoptic":
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
                model.model.metadata = metadata
                model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
                outputs = model.forward(batch_inputs)
 
                visual = Visualizer(image_ori, metadata=metadata)
                pano_seg = outputs[-1]['panoptic_seg'][0].cpu()
                pano_seg_info = outputs[-1]['panoptic_seg'][1]
 
                # Remap category IDs for COCO-style categories
                for i in range(len(pano_seg_info)):
                    seg_info = pano_seg_info[i]
                    if seg_info['isthing']:
                        seg_info['category_id'] = metadata.thing_dataset_id_to_contiguous_id[seg_info['category_id']]
                        class_name = metadata.thing_classes[seg_info['category_id']]
                    else:
                        seg_info['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[seg_info['category_id']]
                        class_name = metadata.stuff_classes[seg_info['category_id']]
                    results["objects"][class_name] = results["objects"].get(class_name, 0) + 1
 
                results["total_count"] = sum(results["objects"].values())
                results["panoptic_seg_map"] = pano_seg.numpy().tolist()
                demo = visual.draw_panoptic_seg(pano_seg, pano_seg_info)
 
                seg_output_path = os.path.join(output_root, "PanopticSeg", os.path.basename(image_pth))
 
                # ---- Save COCO-style annotation ----
                if not os.path.exists(os.path.join(output_root, "PanopticSegPNG")):
                    os.makedirs(os.path.join(output_root, "PanopticSegPNG"))
 
                file_name = f"{image_id:012d}.png"
                png_path = os.path.join(output_root, "PanopticSegPNG", file_name)
                Image.fromarray(pano_seg.byte().numpy()).save(png_path)
 
                # Save COCO-style panoptic annotation JSON
                annotation = generate_panoptic_annotations(pano_seg, pano_seg_info, image_id, os.path.join(output_root, "PanopticSegPNG"))
                coco_output["annotations"].append(annotation)
                coco_output["images"].append({
                    "id": image_id,
                    "file_name": os.path.basename(image_pth),
                    "height": height,
                    "width": width
                })
		
                image_id += 1
                segment_id += len(pano_seg_info)
 
            # Save segmentation image
            if demo is not None:
                if not os.path.exists(os.path.dirname(seg_output_path)):
                    os.makedirs(os.path.dirname(seg_output_path))
                demo.save(seg_output_path)
 
    return results, image_id, segment_id
 
def process_folder(input_folder, output_root, model, transform, metadata_dict):
    os.makedirs(output_root, exist_ok=True)
    coco_json = {"images": [], "annotations": [], "categories": []}
    for idx, name in enumerate(thing_classes + stuff_classes):
        coco_json["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "thing" if idx < len(thing_classes) else "stuff"
        })
        
    results = {}
    image_id = 1
    segment_id = 1
 
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_pth = os.path.join(input_folder, filename)
            print(f"Processing {image_pth}...")
            results, image_id, segment_id = process_image(
                image_pth, output_root, model, transform, metadata_dict,
                coco_json, image_id, segment_id
            )
 
    json_path = os.path.join(output_root, "panoptic_coco_output.json")
    with open(json_path, "w") as f:
        json.dump(coco_json, f, indent=4)
    print(f"COCO panoptic-style JSON saved to {json_path}")
    
    json_path = os.path.join(output_root, "results.json")
    with open(json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"JSON results saved to {json_path}")
 
def main(args=None):
    opt, _ = load_opt_command(args)
    model = setup_model(opt)
    metadata_dict = setup_metadata()
    input_folder = opt.get("input_folder", "input/")
    output_root = opt.get("output_folder", "output/")
    transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC)])
    process_folder(input_folder, output_root, model, transform, metadata_dict)
 
if __name__ == "__main__":
    main()
