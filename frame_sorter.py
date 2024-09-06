import os
import argparse
import numpy as np
import cv2
import torch
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Optional
import shutil
def debug_print(debug: bool, message: str) -> None:
    """Print debug information if debug mode is enabled."""
    if debug:
        print(f"DEBUG: {message}")

def load_frame_paths(directory: str, debug: bool = False) -> List[str]:
    """Load frame file paths and sort them by timestamp."""
    frames = [f for f in os.listdir(directory) if f.endswith('.png')]
    frame_paths = sorted([os.path.join(directory, f) for f in frames], key=lambda x: int(os.path.basename(x).split('.')[0]))
    debug_print(debug, f"Found {len(frame_paths)} frames.")
    return frame_paths

def load_huggingface_model(model_name: str = 'google/vit-base-patch16-224-in21k', debug: bool = False) -> Tuple[AutoFeatureExtractor, AutoModel]:
    """Load Hugging Face model and feature extractor."""
    debug_print(debug, f"Loading Hugging Face model: {model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return feature_extractor, model

def extract_frame_features(frame_path: str, feature_extractor: Optional[AutoFeatureExtractor] = None, model: Optional[AutoModel] = None, debug: bool = False) -> np.ndarray:
    """Extract features from a single image frame using a Hugging Face model."""
    if not feature_extractor or not model:
        feature_extractor, model = load_huggingface_model(debug=debug)
    image = cv2.imread(frame_path)

    # Convert image into the format expected by ViT
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the feature vector and flatten it
    feature_vector = outputs.last_hidden_state.mean(dim=1).flatten().numpy()
    debug_print(debug, f"Extracted and flattened features for frame: {frame_path}")

    return feature_vector

def extract_frame_features_and_classify(frame_path: str, feature_extractor: AutoFeatureExtractor, model: AutoModel, debug: bool = False) -> Tuple[np.ndarray, int]:
    """Extract features from a single image frame and classify it using a Hugging Face model."""
    image = cv2.imread(frame_path)

    # Convert image into the format expected by ViT
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the feature vector and predicted class
    feature_vector = outputs.last_hidden_state.mean(dim=1).numpy()
    predicted_class = outputs.logits.argmax().item()

    debug_print(debug, f"Extracted features and predicted class for frame: {frame_path}")

    return feature_vector, predicted_class

def calculate_frame_features_and_classify(frame_paths: List[str], model_name: str = 'facebook/timesformer-base-finetuned-k400', debug: bool = False) -> Tuple[np.ndarray, List[int]]:
    """Calculate features and classify all frames using Hugging Face's model."""
    feature_extractor, model = load_huggingface_model(model_name, debug)
    features = []
    predicted_classes = []
    for frame_path in frame_paths:
        feature_vector, predicted_class = extract_frame_features_and_classify(frame_path, feature_extractor, model, debug)
        features.append(feature_vector)
        predicted_classes.append(predicted_class)
    debug_print(debug, f"Extracted features and predicted classes for {len(frame_paths)} frames.")
    return np.array(features), predicted_classes

def save_frames_with_labels_to_directories(frame_paths: List[str], labels: List[int], predicted_classes: List[int], output_dir: str, debug: bool = False) -> None:
    """Save frames into directories based on their labels (group) and predicted classes."""
    unique_labels = set(labels)
    for label in unique_labels:
        group_dir = os.path.join(output_dir, f"video_{label}")
        os.makedirs(group_dir, exist_ok=True)
        for i, (frame_path, predicted_class) in enumerate(zip(frame_paths, predicted_classes)):
            if labels[i] == label:
                frame_name = os.path.basename(frame_path)
                dest_path = os.path.join(group_dir, f"{frame_name}_{predicted_class}.png")
                #os.rename(frame_path, dest_path)
                shutil.copy(frame_path, dest_path)
                debug_print(debug, f"Moved {frame_path} to {dest_path}")

def group_frames_by_features(features: np.ndarray, debug: bool = False) -> np.ndarray:
    """Cluster frames based on their extracted features."""
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(features)
    debug_print(debug, "Performed clustering on extracted features.")
    return clustering.labels_

def calculate_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray, debug: bool = False) -> float:
    """Calculate optical flow between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    debug_print(debug, f"Optical flow magnitude between frames: {magnitude}")
    return np.mean(magnitude)

def refine_groups_with_optical_flow(groups: List[List[str]], optical_flow_threshold: float, debug: bool = False) -> List[List[str]]:
    """Refine frame groups by calculating optical flow between consecutive frames."""
    refined_groups = []

    for group in groups:
        if len(group) == 1:
            refined_groups.append(group)
            continue

        refined_group = [group[0]]
        prev_frame = cv2.imread(group[0])

        for i in range(1, len(group)):
            curr_frame = cv2.imread(group[i])
            flow_magnitude = calculate_optical_flow(prev_frame, curr_frame, debug)

            if flow_magnitude <= optical_flow_threshold:
                refined_group.append(group[i])
            else:
                refined_groups.append(refined_group)
                refined_group = [group[i]]

            prev_frame = curr_frame

        if refined_group:
            refined_groups.append(refined_group)

    debug_print(debug, f"Refined groups based on optical flow.")
    return refined_groups

def save_frames_to_directories_by_labels(frame_paths: List[str], labels: np.ndarray, output_dir: str, debug: bool = False) -> None:
    """Save frames into directories based on their labels (group)."""
    unique_labels = set(labels)
    for label in unique_labels:
        group_dir = os.path.join(output_dir, f"video_{label}")
        os.makedirs(group_dir, exist_ok=True)
        for i, frame_path in enumerate(frame_paths):
            if labels[i] == label:
                frame_name = os.path.basename(frame_path)
                dest_path = os.path.join(group_dir, frame_name)
                os.rename(frame_path, dest_path)
                debug_print(debug, f"Moved {frame_path} to {dest_path}")

def main() -> None:
    # Set up argument parser to accept flags
    parser = argparse.ArgumentParser(description="Sort video frames into groups based on similarity.")
    parser.add_argument('-i', '--inputDir', required=True, help="Directory containing the video frames.")
    parser.add_argument('-o', '--outputDir', required=True, help="Directory where the sorted frames will be saved.")
    parser.add_argument('-f', '--opticalFlow', action='store_true', help="Enable optical flow refinement for grouping.")
    parser.add_argument('-m', '--modelName', default='facebook/timesformer-base-finetuned-k400', help="Hugging Face model name for feature extraction.")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode to print additional information.")
    args = parser.parse_args()

    model_name: str = args.modelName or "google/vit-base-patch16-224-in21k"
    debug: bool = args.debug or False

    out_dir: str = args.outputDir
    if not out_dir:
        raise ValueError("Output directory must be specified.")

    in_dir: str = args.inputDir
    if not in_dir:
        raise ValueError("Input directory must be specified.")

    # Load frame paths
    frame_paths: List[str] = load_frame_paths(in_dir, debug=args.debug)
    print(f"Loaded {len(frame_paths)} frames from {args.inputDir}")

    # Step 1: Calculate features using Hugging Face model
    features: List[np.ndarray] = []
    for frame_path in frame_paths:
        feature = extract_frame_features(frame_path, model=model_name, debug=debug)
        features.append(feature)

    # Step 2: Group frames by features
    labels: np.ndarray = group_frames_by_features(np.array(features), debug=debug)

    # Step 3: Refine groups using optical flow if enabled
    if args.opticalFlow:
        print("Refining groups using optical flow...")
        labels = refine_groups_with_optical_flow(labels, optical_flow_threshold=1.0, debug=debug)

    # Step 4: Save grouped frames to the output directory
    save_frames_to_directories_by_labels(frame_paths, labels, out_dir, debug=debug)
    print(f"Frames have been sorted and saved to {out_dir}.")

if __name__ == "__main__":
    main()
