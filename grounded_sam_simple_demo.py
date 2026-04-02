import os
import cv2
import argparse
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

# 設定命令列參數解析器
parser = argparse.ArgumentParser(description="Grounded-SAM 圖片分割工具")
parser.add_argument(
    "-i", "--image",
    type=str,
    nargs="+",
    required=True, 
    help="請輸入一張或多張要處理的圖片路徑 (例如: test_images/a.png test_images/b.png)"
)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

SOURCE_IMAGE_PATHS = args.image
CLASSES = ["French's Mustard bottle", "Tomato Soup can", "Plastic banana", "Clamp", "Orange Drill"] 
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.3
NMS_THRESHOLD = 0.8

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()


def process_image(source_image_path: str) -> str:
    dir_name = os.path.dirname(source_image_path)
    filename = os.path.basename(source_image_path)
    base_name, _ = os.path.splitext(filename)

    if not os.path.exists(source_image_path):
        raise FileNotFoundError(f"找不到檔案或目錄 '{source_image_path}'")

    image = cv2.imread(source_image_path)
    if image is None:
        raise ValueError(f"檔案存在但無法讀取為圖片格式 ({source_image_path})")

    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    print(f"[{source_image_path}] Before NMS: {len(detections.xyxy)} boxes")
    if len(detections.xyxy) > 0:
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
    else:
        detections.mask = np.empty((0,), dtype=bool)

    print(f"[{source_image_path}] After NMS: {len(detections.xyxy)} boxes")

    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    if dir_name == "":
        sam_output_path = f"{base_name}_grounded_sam_simple.jpg"
    else:
        sam_output_path = os.path.join(dir_name, f"{base_name}_grounded_sam_simple.jpg")

    cv2.imwrite(sam_output_path, annotated_image)
    print(f"Saved: {sam_output_path}")

    return annotated_image


annotated_results = []
for source_image_path in SOURCE_IMAGE_PATHS:
    try:
        annotated_results.append((source_image_path, process_image(source_image_path)))
    except (FileNotFoundError, ValueError) as error:
        print(f"錯誤：{error}")

if not annotated_results:
    raise SystemExit(1)

if len(annotated_results) == 1:
    source_image_path, annotated_image = annotated_results[0]
    print(f"正在開啟圖片視窗：{source_image_path}，請關閉視窗以結束程式...")
    image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Grounded-SAM Result")
    plt.show()
else:
    print(f"共處理 {len(annotated_results)} 張圖片，結果已儲存到各自輸入圖片所在資料夾。")
