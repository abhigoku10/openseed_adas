#!/bin/bash
set -e

echo "Compiling CUDA extensions"
cd openseed/body/encoder/ops
sh make.sh
cd ../../../..

#Parameters
TASK=${1:-image} # image | video
INPUT_FOLDER=${2:-input} # default sub-folder inside repo
OUTPUT_FOLDER=${3:-output}
CONFIG_FILE="configs/openseed/openseed_swinl_lang_decouple.yaml"
WEIGHT_FILE="weights/openseed_swinl_pano_sota.pt"

echo "OpenSeeD docker — task=$TASK input=$INPUT_FOLDER output=$OUTPUT_FOLDER"

# Create I/O dirs
mkdir -p "$INPUT_FOLDER" "$OUTPUT_FOLDER"

if [[ "$TASK" == "image" ]]; then
  python3 openseed_inference_script.py evaluate \
    --conf_files "$CONFIG_FILE" \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --overrides WEIGHT "$WEIGHT_FILE"
    

elif [[ "$TASK" == "video" ]]; then
  python3 openseed_inference_video.py evaluate \
    --conf_files "$CONFIG_FILE" \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --overrides WEIGHT "$WEIGHT_FILE"
    
else
  echo "Unknown task '$TASK' — use 'image' or 'video'"
  exit 1
fi
