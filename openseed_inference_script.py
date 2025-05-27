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

thing_classes = ["car", "person", "traffic light", "truck", "motorcycle", "bicycle", "sign board", "bus"]
stuff_classes = ['building', 'sky', 'street', 'tree', 'rock', 'sidewalk', 'house', 'mountain', 'grass', 'stone', 'road', 'lane']
 
def setup_model(opt):
    pretrained_pth = os.path.join(opt['WEIGHT'])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    return model
 
def setup_metadata():
    # Instance Segmentation Metadata    
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
    thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}
    instance_metadata = MetadataCatalog.get("instance").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id
    )
 
    # Semantic Segmentation Metadata
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x: x for x in range(len(stuff_classes))}
    semantic_metadata = MetadataCatalog.get("semantic").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id
    )
 
    # Panoptic Segmentation Metadata
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes): x for x in range(len(stuff_classes))}
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
 
def process_image(image_pth, output_root, model, transform, metadata_dict):
    image_ori = Image.open(image_pth).convert("RGB")
    width, height = image_ori.size
    image = transform(image_ori)
    image = np.asarray(image)
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
 
    batch_inputs = [{'image': images, 'height': height, 'width': width}]
    
    results = {
        "image": os.path.basename(image_pth),
        "objects": {},
        "total_count": 0,
    }
    
    #detections = {}
     
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
                
                # # Extract JSON data
                # for i in range(len(inst_seg)):
                #     class_name = metadata.thing_classes[inst_seg.pred_classes[i]]
                #     bbox = inst_seg.pred_boxes.tensor[i].tolist()
                #     conf = float(inst_seg.scores[i])
                #     if class_name not in detections:
                #         detections[class_name] = []
                #     detections[class_name].append({"bbox": bbox, "conf": conf})
            
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
	    	
                pano_seg = outputs[-1]['panoptic_seg'][0]
                pano_seg_info = outputs[-1]['panoptic_seg'][1]
 
                for i in range(len(pano_seg_info)):
                    if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                        pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                        class_name = metadata.thing_classes[pano_seg_info[i]['category_id']]
                    else:
                        pano_seg_info[i]['isthing'] = False
                        pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                        class_name = metadata.stuff_classes[pano_seg_info[i]['category_id']]

                    results["objects"][class_name] = results["objects"].get(class_name, 0) + 1
 
                demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info)
                seg_output_path = os.path.join(output_root, "PanopticSeg", os.path.basename(image_pth))
                
                results["total_count"] = sum(results["objects"].values())
                results["panoptic_seg_map"] = pano_seg.cpu().numpy().tolist()

 
            if not os.path.exists(os.path.dirname(seg_output_path)):
                os.makedirs(os.path.dirname(seg_output_path))
            
            demo.save(seg_output_path)
 
    # results["num_objects"] = sum(len(v) for v in detections.values())
    # results.update(detections)
 
    return results
 
def process_folder(input_folder, output_root, model, transform, metadata_dict):
    all_results = []
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_pth = os.path.join(input_folder, filename)
            print(f"Processing {image_pth}...")
            results = process_image(image_pth, output_root, model, transform, metadata_dict)
            all_results.append(results)
 
    json_path = os.path.join(output_root, "results.json")
    with open(json_path, "w") as json_file:
        json.dump(all_results, json_file, indent=4)
    
    print(f"JSON results saved to {json_path}")
 
def main(args=None):
    opt, _ = load_opt_command(args)
    model = setup_model(opt)
    metadata_dict = setup_metadata()
 
    input_folder = opt.get("input_folder", "input/")
    output_root = opt.get("output_folder", "output/")
    
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=Image.BICUBIC)
    ])
 
    process_folder(input_folder, output_root, model, transform, metadata_dict)
 
if __name__ == "__main__":
    main()
