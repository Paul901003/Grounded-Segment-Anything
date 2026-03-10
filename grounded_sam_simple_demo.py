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
    required=True, 
    help="請輸入要處理的圖片路徑 (例如: test_images/robot.png)"
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

# 將 SOURCE_IMAGE_PATH 替換為我們從命令列接收到的參數
SOURCE_IMAGE_PATH = args.image
CLASSES = ["box", "can", "bottle"] 
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# 取得輸入檔案的路徑目錄與名稱
dir_name = os.path.dirname(SOURCE_IMAGE_PATH)
filename = os.path.basename(SOURCE_IMAGE_PATH)
base_name, _ = os.path.splitext(filename)

if not os.path.exists(SOURCE_IMAGE_PATH):
    print(f"錯誤：找不到檔案或目錄 '{SOURCE_IMAGE_PATH}'")
    exit(1) # 加上 1 代表程式異常結束

image = cv2.imread(SOURCE_IMAGE_PATH)
if image is None:
    print(f"錯誤：檔案存在但無法讀取為圖片格式 ({SOURCE_IMAGE_PATH})")
    exit(1)

# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# NMS post process
print(f"Before NMS: {len(detections.xyxy)} boxes")
nms_idx = torchvision.ops.nms(
    torch.from_numpy(detections.xyxy), 
    torch.from_numpy(detections.confidence), 
    NMS_THRESHOLD
).numpy().tolist()

detections.xyxy = detections.xyxy[nms_idx]
detections.confidence = detections.confidence[nms_idx]
detections.class_id = detections.class_id[nms_idx]

print(f"After NMS: {len(detections.xyxy)} boxes")

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

# convert detections to masks
detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

# annotate image with detections (包含 Mask 與 Box)
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _
    in detections]

annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# 組合目錄路徑與新檔名，確保儲存在同一資料夾
# 若輸入路徑沒有目錄 (例如直接輸入 img.png)，dir_name 會是空字串，這裡做個小處理避免存檔路徑出錯
if dir_name == "":
    sam_output_path = f"{base_name}_grounded_sam_simple.jpg"
else:
    sam_output_path = os.path.join(dir_name, f"{base_name}_grounded_sam_simple.jpg")

cv2.imwrite(sam_output_path, annotated_image)
print(f"Saved: {sam_output_path}")

# 直接在螢幕上顯示結果 (改用 matplotlib)
print("正在開啟圖片視窗，請關閉視窗以結束程式...")

# 將 BGR 轉換為 RGB 以便 matplotlib 正確顯示顏色
image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))  # 可以自行調整顯示視窗的大小
plt.imshow(image_rgb)
plt.axis('off')               # 隱藏 XY 座標軸
plt.title("Grounded-SAM Result")
plt.show()                    # 顯示圖片，程式會停在這裡直到你關閉視窗