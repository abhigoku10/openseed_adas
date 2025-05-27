import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.arguments import load_opt_command
from detectron2.utils.colormap import random_color
 
# Define your panoptic classes
thing_classes = ["car", "person", "traffic light", "truck", "motorcycle", "bicycle", "sign board", "road", "bus", "lane"]
stuff_classes = ['building', 'sky', 'street', 'tree', 'rock', 'sidewalk', 'house', 'mountain', 'grass', 'stone']
 
def setup_model(opt):
    pretrained_pth = os.path.join(opt['WEIGHT'])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    return model
 
def setup_metadata():
    # Setup Metadata for panoptic segmentation
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(stuff_classes))]
 
    panoptic_metadata = MetadataCatalog.get("panoptic").set(
        thing_classes=thing_classes,
        thing_colors=thing_colors,
        stuff_classes=stuff_classes,
        stuff_colors=stuff_colors,
        thing_dataset_id_to_contiguous_id={x: x for x in range(len(thing_classes))},
        stuff_dataset_id_to_contiguous_id={x+len(thing_classes): x for x in range(len(stuff_classes))}
    )
 
    return panoptic_metadata
 
def process_video(video_path, output_root, model, transform, metadata):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_output_dir = os.path.join(output_root, video_name)
    os.makedirs(frame_output_dir, exist_ok=True)
 
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
 
    frame_idx = 0
    frame_paths = []
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        image_ori = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_ori_pil = Image.fromarray(image_ori)
        image_resized = transform(image_ori_pil)
        image_np = np.asarray(image_resized)
        images = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()
 
        batch_inputs = [{'image': images, 'height': frame_height, 'width': frame_width}]
        
        with torch.no_grad():
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
            model.model.metadata = metadata
            model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
 
            outputs = model.forward(batch_inputs)
            pano_seg, pano_seg_info = outputs[-1]['panoptic_seg']
            
            for i in range(len(pano_seg_info)):
                if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                    pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                elif pano_seg_info[i]['category_id'] in metadata.stuff_dataset_id_to_contiguous_id.keys():
                    pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                else:
                    print(f"Warning: Unknown category_id {pano_seg_info[i]['category_id']}. Skipping this segment.")
                    pano_seg_info[i]['isthing'] = False
            
            visualizer = Visualizer(np.asarray(image_ori), metadata=metadata)
            vis_output = visualizer.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info)
 
            frame_output_path = os.path.join(frame_output_dir, f"frame_{frame_idx:05d}.png")
            vis_output.save(frame_output_path)
            frame_paths.append(frame_output_path)
            frame_idx += 1
 
    cap.release()
 
    # Create Video
    output_video_path = os.path.join(output_root, f"{video_name}_segmented.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
 
    for path in sorted(frame_paths):
        frame = cv2.imread(path)
        out.write(frame)
 
    out.release()
    print(f"Processed video saved at {output_video_path}")
 

def process_folder(input_folder, output_root, model, transform, metadata):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
 
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(input_folder, filename)
            print(f"Processing video {video_path}...")
            process_video(video_path, output_root, model, transform, metadata)
 
def main(args=None):
    opt, _ = load_opt_command(args)
    model = setup_model(opt)
    metadata = setup_metadata()
 
    #input_folder = "videos/"   # Your input folder with videos
    #output_root = "video_output/"   # Output folder
    input_folder = opt.get("input_folder", "input/")
    output_root = opt.get("output_folder", "output/")
 
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC)  # Resize frames to 512x512
    ])
 
    process_folder(input_folder, output_root, model, transform, metadata)
 
if __name__ == "__main__":
    main()
