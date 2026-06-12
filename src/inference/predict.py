import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def preprocess_image(
    image_path,
    image_size = (128, 128)):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, image_size)

    img_resized = cv2.normalize(
        img_resized, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F
    )
    
    return np.expand_dims(img_resized, axis=0)


def postprocess_prediction(
    pred: np.ndarray,
    threshold: float = 0.5):

    pred = pred.squeeze(0)
    binary = (pred >= threshold).astype(np.uint8)
    
    return binary


def visualize_prediction(
    original_img: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.4):

    if original_img.max() <= 1.0:
        img_vis = (original_img * 255).astype(np.uint8)
    else:
        img_vis = original_img.astype(np.uint8)
    
    num_classes = prediction.shape[-1]
    
    colors = [
        (0, 255, 0),
        (255, 0, 0),
    ]
    
    mask_color = np.zeros_like(img_vis)
    for class_idx in range(num_classes):
        if class_idx < len(colors):
            channel_mask = prediction[:, :, class_idx]
            for c in range(3):
                mask_color[:, :, c] += channel_mask * colors[class_idx][c]
    
    overlay = cv2.addWeighted(img_vis, 1.0, mask_color, alpha, 0)
    result = np.concatenate([img_vis, mask_color, overlay], axis=1)
    
    return result


def save_prediction(
    prediction,
    output_path,
    original_img = None,
    save_mask = False,
    save_visualization = True):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_mask:
        np.save(str(output_path) + "_mask.npy", prediction)
    
    if save_visualization and original_img is not None:
        viz = visualize_prediction(original_img, prediction)
        viz_bgr = cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path) + "_viz.png", viz_bgr)
        


def run(cfg: dict, checkpoint: str, input_source = None):

    model = tf.keras.models.load_model(checkpoint, compile=False)
    
    paths = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    
    images_dir = Path(paths.get("dataset", "dataset")) / "images"
    output_dir = Path(paths.get("logs", "logs")) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_size = tuple(data_cfg.get("image_size", (128, 128)))
    num_classes = 2

    if input_source is None:
        input_files = list((images_dir).glob("*.jpg"))
        if not input_files:
            input_files = list((images_dir).glob("*.png"))
    elif isinstance(input_source, str):
        src_path = Path(input_source)
        if src_path.is_file():
            input_files = [src_path]
        elif src_path.is_dir():
            input_files = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))
        else:
            raise ValueError(f"Неверный input_source: {input_source}")
    else:
        input_files = [Path(p) for p in input_source]
    
    if not input_files:
        print("Не найдено изображений для инференса")
        return
    
    print(f"Найдено {len(input_files)} изображений")

    for img_path in input_files:
        try:
            input_tensor = preprocess_image(img_path, image_size)
            pred = model.predict(input_tensor, verbose=0)
            binary_mask = postprocess_prediction(pred)
            original = cv2.imread(str(img_path))
            original = cv2.resize(original, (128, 128))
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original is not None else None
            stem = img_path.stem
            save_prediction(
                prediction=binary_mask,
                output_path=output_dir / stem,
                original_img=original_rgb,
                save_mask=cfg['inference']['save_mask'],
                save_visualization=cfg['inference']['save_visualization'],
            )
            
        except Exception as e:
            print(f"Ошибка при обработке {img_path.name}: {e}")
            continue

def run_single(cfg: dict, checkpoint: str, image_path: str, return_result: bool = False):

    model = tf.keras.models.load_model(checkpoint, compile=False)
    
    data_cfg = cfg.get("data", {})
    image_size = tuple(data_cfg.get("image_size", (128, 128)))
    
    input_tensor = preprocess_image(image_path, image_size)
    
    pred = model.predict(input_tensor, verbose=0)
    binary_mask = postprocess_prediction(pred)
    
    if return_result:
        original = cv2.imread(str(image_path))
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if original is not None else None
        viz = visualize_prediction(original_rgb, binary_mask) if original_rgb is not None else None
        return binary_mask, viz
    
    output_path = Path(cfg.get("paths", {}).get("logs", "logs")) / "predictions" / Path(image_path).stem
    save_prediction(binary_mask, output_path, original_rgb)
    
    return None