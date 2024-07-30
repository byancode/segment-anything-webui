import argparse
import torch
from PIL import Image
import io
import base64
import numpy as np
import cv2
from segment_anything import SamPredictor

# Cargar el modelo de Segment Anything
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamPredictor.from_pretrained("model/sam_vit_h_4b8939.pth")
model.to(device)
model.eval()

def load_image(file_path):
    with open(file_path, 'rb') as f:
        image = Image.open(f).convert("RGB")
    return image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contours_to_svg(contours):
    svg_paths = []
    for contour in contours:
        path = "M " + " L ".join(f"{point[0][0]},{point[0][1]}" for point in contour)
        svg_paths.append(path + " Z")
    return svg_paths

def segment_click(image_path, x, y):
    image = load_image(image_path)
    input_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)
    click_point = torch.tensor([[x, y]]).float().to(device)

    with torch.no_grad():
        predictions = model.segment_click(input_tensor, click_point)

    return process_predictions(predictions)

def segment_box(image_path, x_min, y_min, x_max, y_max):
    image = load_image(image_path)
    input_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)
    box_coords = torch.tensor([[x_min, y_min, x_max, y_max]]).float().to(device)

    with torch.no_grad():
        predictions = model.segment_box(input_tensor, box_coords)

    return process_predictions(predictions)

def segment_anything(image_path):
    image = load_image(image_path)
    input_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model.segment_anything(input_tensor)

    return process_predictions(predictions)

def segment_sem(image_path):
    image = load_image(image_path)
    input_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model.segment_sem(input_tensor)

    return process_predictions(predictions)

def process_predictions(predictions):
    segmented_images = []
    boundaries = []
    clip_paths = []
    for prediction in predictions:
        mask = prediction.cpu().numpy().astype(np.uint8)
        contours = get_contours(mask)
        svg_paths = contours_to_svg(contours)

        output_image = Image.fromarray(mask * 255)
        img_str = image_to_base64(output_image)
        segmented_images.append(img_str)
        boundaries.append(contours)
        clip_paths.append(svg_paths)

    return {
        "segmented_images": segmented_images,
        "boundaries": boundaries,
        "clip_paths": clip_paths
    }

def main():
    parser = argparse.ArgumentParser(description="Segment Anything CLI")
    subparsers = parser.add_subparsers(dest="mode", help="Segmentation mode")

    parser_click = subparsers.add_parser("click", help="Segment using click mode")
    parser_click.add_argument("image_path", type=str, help="Path to the input image")
    parser_click.add_argument("x", type=int, help="X coordinate for click")
    parser_click.add_argument("y", type=int, help="Y coordinate for click")

    parser_box = subparsers.add_parser("box", help="Segment using box mode")
    parser_box.add_argument("image_path", type=str, help="Path to the input image")
    parser_box.add_argument("x_min", type=int, help="Minimum X coordinate for box")
    parser_box.add_argument("y_min", type=int, help="Minimum Y coordinate for box")
    parser_box.add_argument("x_max", type=int, help="Maximum X coordinate for box")
    parser_box.add_argument("y_max", type=int, help="Maximum Y coordinate for box")

    parser_anything = subparsers.add_parser("anything", help="Segment using segment anything mode")
    parser_anything.add_argument("image_path", type=str, help="Path to the input image")

    parser_sem = subparsers.add_parser("sem", help="Segment using semantic segmentation mode")
    parser_sem.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()

    if args.mode == "click":
        result = segment_click(args.image_path, args.x, args.y)
    elif args.mode == "box":
        result = segment_box(args.image_path, args.x_min, args.y_min, args.x_max, args.y_max)
    elif args.mode == "anything":
        result = segment_anything(args.image_path)
    elif args.mode == "sem":
        result = segment_sem(args.image_path)
    else:
        parser.print_help()
        return

    print(result)

if __name__ == "__main__":
    main()
