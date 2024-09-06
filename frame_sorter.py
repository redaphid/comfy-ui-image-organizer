import os
import argparse
import numpy as np
import cv2
import torch
import shutil
import sys
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.cluster import DBSCAN, KMeans
from typing import List, Tuple, Optional

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

def load_huggingface_model(model_name: str, debug: bool = False) -> Tuple[AutoFeatureExtractor, AutoModel]:
    """Load Hugging Face model and feature extractor."""
    debug_print(debug, f"Loading Hugging Face model: {model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return feature_extractor, model

def extract_frame_features(frame_path: str, feature_extractor: AutoFeatureExtractor, model: AutoModel, debug: bool = False) -> np.ndarray:
    """Extract features from a single image frame using a Hugging Face model."""
    image = cv2.imread(frame_path)
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the feature vector and flatten it for clustering
    feature_vector = outputs.last_hidden_state.mean(dim=1).flatten().numpy()
    return feature_vector

def group_frames_by_features_dbscan(features: np.ndarray, eps: float = 0.5, min_samples: int = 2, debug: bool = False) -> np.ndarray:
    """Cluster frames based on their extracted features using DBSCAN."""
    debug_print(debug, f"Running DBSCAN with eps={eps} and min_samples={min_samples}")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

    # Log the clustering labels
    unique_labels = set(clustering.labels_)
    debug_print(debug, f"DBSCAN found {len(unique_labels)} groups (including noise).")
    if -1 in unique_labels:
        noise_points = list(clustering.labels_).count(-1)
        debug_print(debug, f"DBSCAN classified {noise_points} points as noise (label -1).")

    return clustering.labels_

def group_frames_by_features_kmeans(features: np.ndarray, num_groups: int, debug: bool = False) -> np.ndarray:
    """Cluster frames using KMeans clustering."""
    debug_print(debug, f"Running KMeans with {num_groups} clusters.")
    clustering = KMeans(n_clusters=num_groups).fit(features)

    # Log the clustering labels
    unique_labels = set(clustering.labels_)
    debug_print(debug, f"KMeans created {len(unique_labels)} clusters.")

    return clustering.labels_

def save_frames_to_directories_by_labels(frame_paths: List[str], labels: np.ndarray, output_dir: str, debug: bool = False) -> None:
    """Save frames into directories based on their labels (group)."""
    unique_labels = set(labels)
    group_file_count = {label: labels.tolist().count(label) for label in unique_labels}

    for label in unique_labels:
        group_dir = os.path.join(output_dir, f"video_{label}")
        os.makedirs(group_dir, exist_ok=True)
        for i, frame_path in enumerate(frame_paths):
            if labels[i] == label:
                frame_name = os.path.basename(frame_path)
                dest_path = os.path.join(group_dir, frame_name)
                shutil.copy2(frame_path, dest_path)  # Copy instead of moving
                debug_print(debug, f"Copied {frame_path} to {dest_path}")

    # Log group and file info
    debug_print(debug, f"Found {len(unique_labels)} groups.")
    for label, count in group_file_count.items():
        print(f"Group {label}: {count} files.")

def update_progress_bar(total_frames: int, current_frame: int) -> None:
    """Display a progress bar or status update for frame processing."""
    sys.stdout.write(f"\rProcessed {current_frame}/{total_frames} frames")
    sys.stdout.flush()

def main() -> None:
    """Main function to process video frames and group them by similarity."""
    parser = argparse.ArgumentParser(description="Sort video frames into groups based on similarity.")
    parser.add_argument('-i', '--inputDir', required=True, help="Directory containing the video frames.")
    parser.add_argument('-o', '--outputDir', required=True, help="Directory where the sorted frames will be saved.")
    parser.add_argument('-m', '--modelName', default='google/vit-base-patch16-224-in21k', help="Hugging Face model name for feature extraction.")
    parser.add_argument('-n', '--numGroups', type=int, default=None, help="Number of groups (clusters) for KMeans.")
    parser.add_argument('-e', '--eps', type=float, default=0.5, help="DBSCAN epsilon (max distance between samples).")
    parser.add_argument('--minSamples', type=int, default=2, help="DBSCAN minimum samples for a cluster.")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode to print additional information.")
    args = parser.parse_args()

    # Load frame paths
    frame_paths: List[str] = load_frame_paths(args.inputDir, debug=args.debug)
    total_frames = len(frame_paths)
    print(f"Loaded {total_frames} frames from {args.inputDir}")

    # Load Hugging Face model and feature extractor
    feature_extractor, model = load_huggingface_model(args.modelName, debug=args.debug)

    # Extract features for all frames and update progress
    features: List[np.ndarray] = []
    for i, frame_path in enumerate(frame_paths):
        feature = extract_frame_features(frame_path, feature_extractor, model, debug=args.debug)
        features.append(feature)
        update_progress_bar(total_frames, i + 1)  # Update progress bar after each frame
    print()  # To ensure the final newline after progress bar

    # Perform clustering
    if args.numGroups:
        labels = group_frames_by_features_kmeans(np.array(features), args.numGroups, debug=args.debug)
    else:
        labels = group_frames_by_features_dbscan(np.array(features), eps=args.eps, min_samples=args.minSamples, debug=args.debug)

    # Save frames into groups and log group sizes
    save_frames_to_directories_by_labels(frame_paths, labels, args.outputDir, debug=args.debug)

    print(f"Frames have been grouped and saved to {args.outputDir}.")

if __name__ == "__main__":
    main()
