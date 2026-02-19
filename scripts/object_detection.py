"""
scripts/object_detection.py
Runs YOLOv8 object detection on an image using Ultralytics.
Usage:
  python object_detection.py --image sample.jpg
"""
import argparse
import numpy as np
from PIL import Image

def detect_image(image_path, model_name='yolov8n.pt', save=False, conf=0.25):
    """
    Run YOLOv8 object detection on the given image path.

    Args:
        image_path (str): Path to the input image.
        model_name (str): YOLOv8 model weights to use (e.g. 'yolov8n.pt').
        save (bool): Whether to save the annotated image to disk.
        conf (float): Confidence threshold for detections.

    Returns:
        np.ndarray: Annotated image as a NumPy array (RGB).
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.predict(source=image_path, conf=conf, save=save, verbose=False)

    # results[0].plot() returns a BGR numpy array — convert to RGB
    annotated_bgr = results[0].plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]
    return annotated_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    annotated = detect_image(args.image, model_name=args.model, conf=args.conf)
    out_path = 'detection_output.jpg'
    Image.fromarray(annotated).save(out_path)
    print(f"Annotated image saved to {out_path}")
