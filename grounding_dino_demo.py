from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "test_images/robot_view_sementation_test.png"
TEXT_PROMPT = "Box. Can. Robotic arm. Plane."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
FP16_INFERENCE = True

image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

if FP16_INFERENCE:
    image = image.half()
    model = model.half()

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)

annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)

input_path = Path(IMAGE_PATH)
output_path = input_path.with_name(f"{input_path.stem}_groundingdino{input_path.suffix}")

cv2.imwrite(str(output_path), annotated_frame)
print(f"Saved to: {output_path}")