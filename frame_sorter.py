import os
import argparse
import numpy as np
import cv2
import torch
import shutil
import sys
import re
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.cluster import DBSCAN, KMeans
from typing import List, Tuple, Optional

from datetime import datetime, timezone
MAX_FILE_SIZE_MB = 10  # Maximum file size in megabytes

def debug_print(debug: bool, message: str) -> None:
    """Print debug information if debug mode is enabled."""
    if debug:
        print(f"DEBUG: {message}")

def load_frame_paths(directory: str, limit: Optional[int] = None, debug: bool = False) -> List[str]:
    """Load frame file paths, sort them by file creation time, and apply the optional limit."""
    frames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

    # Sort by file creation time (use `st_birthtime`, or `st_mtime` if creation time is not available)
    frame_paths = sorted(frames, key=lambda x: os.stat(x).st_birthtime)

    # Apply limit if provided
    if limit:
        frame_paths = frame_paths[:limit]

    debug_print(debug, f"Found {len(frame_paths)} frames.")
    return frame_paths

def load_huggingface_model(model_name: str, debug: bool = False) -> Tuple[AutoFeatureExtractor, AutoModel]:
    """Load Hugging Face model and feature extractor."""
    debug_print(debug, f"Loading Hugging Face model: {model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return feature_extractor, model

def extract_frame_features(frame_path: str, feature_extractor: AutoFeatureExtractor, model: AutoModel, debug: bool = False) -> Optional[np.ndarray]:
    """Extract features from a single image frame using a Hugging Face model, ignoring files larger than 10 MB."""

    # Check the file size and ignore files larger than 10MB
    file_size_mb = os.path.getsize(frame_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        debug_print(debug, f"Skipping {frame_path}: File size {file_size_mb:.2f} MB exceeds 10 MB.")
        return None

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

    # Log the clustering process
    unique_labels = set(clustering.labels_)
    debug_print(debug, f"DBSCAN found {len(unique_labels)} groups (including noise).")
    if -1 in unique_labels:
        noise_points = list(clustering.labels_).count(-1)
        debug_print(debug, f"DBSCAN classified {noise_points} points as noise (label -1).")

    return clustering.labels_

def group_frames_by_features_kmeans(features: np.ndarray, num_groups: int, debug: bool = False) -> np.ndarray:
    """Cluster frames using KMeans clustering."""
    debug_print(debug, f"Running KMeans with {num_groups} clusters.")
    kmeans = KMeans(n_clusters=num_groups, n_init=10, verbose=1 if debug else 0).fit(features)

    # Log the clustering result
    unique_labels = set(kmeans.labels_)
    debug_print(debug, f"KMeans created {len(unique_labels)} clusters.")

    return kmeans.labels_

def save_checkpoint(features: List[np.ndarray], output_dir: str, iteration: int, debug: bool = False) -> None:
    """Save progress at a given iteration to a checkpoint file."""
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration}.npz")
    np.savez_compressed(checkpoint_path, features=features)
    debug_print(debug, f"Saved progress at iteration {iteration} to {checkpoint_path}.")

def save_frames_to_directories_by_labels(frame_paths: List[str], labels: np.ndarray, output_dir: str, debug: bool = False) -> None:
    """Save frames into directories based on their labels (group), ordered by group size and sorted by creation time."""
    unique_labels = set(labels)

    # Count files in each group
    group_file_count = {label: labels.tolist().count(label) for label in unique_labels}

    # Sort groups by the number of files in descending order
    sorted_groups = sorted(group_file_count.items(), key=lambda x: x[1], reverse=True)

    # Handle noise frames (-1 label)
    if -1 in unique_labels:
        noise_frames = [(fp, os.stat(fp).st_birthtime) for i, fp in enumerate(frame_paths) if labels[i] == -1]
        noise_dir = os.path.join(output_dir, "noise_frames")
        os.makedirs(noise_dir, exist_ok=True)

        for frame_path, creation_time in sorted(noise_frames, key=lambda x: x[1]):
            iso_time = datetime.fromtimestamp(creation_time, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
            dest_path = os.path.join(noise_dir, f"{iso_time}.png")
            shutil.copy2(frame_path, dest_path)
            debug_print(debug, f"Copied noise frame {frame_path} to {dest_path}")

        # Remove noise label from unique labels to avoid creating a folder for it
        unique_labels.remove(-1)

    # For each group, save the frames sorted by file creation time
    for rank, (label, _) in enumerate(sorted_groups, 1):
        if label == -1:
            continue  # Skip the noise label as it's handled separately

        # Collect frame paths and sort by creation time
        group_frames = [(fp, os.stat(fp).st_birthtime) for i, fp in enumerate(frame_paths) if labels[i] == label]
        group_frames_sorted = sorted(group_frames, key=lambda x: x[1])  # Sort by creation time

        # Create output directory for the group
        group_dir = os.path.join(output_dir, f"{rank}_video_{label}")
        os.makedirs(group_dir, exist_ok=True)

        # Copy and rename files using a modified ISO-8601 UTC format for creation time
        for frame_path, creation_time in group_frames_sorted:
            iso_time = datetime.fromtimestamp(creation_time, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
            dest_path = os.path.join(group_dir, f"{iso_time}.png")
            shutil.copy2(frame_path, dest_path)
            debug_print(debug, f"Copied {frame_path} to {dest_path}")

    # Log group and file info
    debug_print(debug, f"Found {len(unique_labels)} groups.")
    for rank, (label, count) in enumerate(sorted_groups, 1):
        print(f"Group {rank} (label {label}): {count} files.")

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
    parser.add_argument('--limit', type=int, help="Limit the number of files to process.")
    parser.add_argument('--checkpointInterval', type=int, default=100, help="Interval for saving checkpoints.")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode to print additional information.")
    args = parser.parse_args()

    # Load frame paths
    frame_paths: List[str] = load_frame_paths(args.inputDir, limit=args.limit, debug=args.debug)
    total_frames = len(frame_paths)
    print(f"Loaded {total_frames} frames from {args.inputDir}")

    # Load Hugging Face model and feature extractor
    feature_extractor, model = load_huggingface_model(args.modelName, debug=args.debug)

    # Extract features for all frames and update progress
    features: List[np.ndarray] = []
    for i, frame_path in enumerate(frame_paths):
        feature = extract_frame_features(frame_path, feature_extractor, model, debug=args.debug)
        if feature is not None:  # Only add features for frames that are not too large
            features.append(feature)

        # Save progress periodically based on the checkpoint interval
        if (i + 1) % args.checkpointInterval == 0:
            save_checkpoint(features, "tmp", i + 1, debug=args.debug)

        # Update progress bar
        update_progress_bar(total_frames, i + 1)
    print()  # To ensure the final newline after progress bar

    # Perform clustering
    if args.numGroups:
        labels = group_frames_by_features_kmeans(np.array(features), args.numGroups, debug=args.debug)
    else:
        labels = group_frames_by_features_dbscan(np.array(features), eps=args.eps, min_samples=args.minSamples, debug=args.debug)

    # Output the number of groups found (excluding noise if using DBSCAN)
    unique_labels = set(labels)
    num_groups = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise (-1) if present
    print(f"Number of groups found: {num_groups}")

    # Save frames into groups and log group sizes
    save_frames_to_directories_by_labels(frame_paths, labels, args.outputDir, debug=args.debug)

    print(f"Frames have been grouped and saved to {args.outputDir}.")

if __name__ == "__main__":
    main()
