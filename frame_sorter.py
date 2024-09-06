import os
import argparse
import numpy as np
import cv2
import torch
import shutil
import sys
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional
from datetime import datetime, timezone

MAX_FILE_SIZE_MB = 10  # Maximum file size in megabytes

def debug_print(debug: bool, message: str) -> None:
    """Print debug information if debug mode is enabled."""
    if debug:
        print(f"DEBUG: {message}")

def load_frame_paths(directory: str, limit: Optional[int] = None, debug: bool = False) -> List[str]:
    """Recursively load all .png frame file paths and apply an optional limit."""
    frames = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                frames.append(os.path.join(root, file))

    # Sort by file creation time (use `st_birthtime`, or `st_mtime` if creation time is not available)
    frame_paths = sorted(frames, key=lambda x: os.stat(x).st_birthtime)

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
    """Extract features from a single image frame using a Hugging Face model."""
    file_size_mb = os.path.getsize(frame_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        debug_print(debug, f"Skipping {frame_path}: File size {file_size_mb:.2f} MB exceeds {MAX_FILE_SIZE_MB} MB.")
        return None

    image = cv2.imread(frame_path)
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).flatten().numpy()

def compute_temporal_similarity(time1: float, time2: float, temporal_weight: float = 0.5) -> float:
    """Compute temporal similarity based on frame creation times."""
    time_diff = abs(time1 - time2)
    return np.exp(-temporal_weight * time_diff)

def evaluate_frame_fit(frame_index: int, features: np.ndarray, window_size: int = 2) -> float:
    """Evaluate how well a frame fits into its temporal neighbors."""
    start = max(0, frame_index - window_size)
    end = min(len(features), frame_index + window_size + 1)

    neighbor_indices = list(range(start, end))
    neighbor_indices.remove(frame_index)

    similarities = []
    for neighbor in neighbor_indices:
        similarity = cosine_similarity([features[frame_index]], [features[neighbor]])[0][0]
        similarities.append(similarity)

    return np.mean(similarities)

def cluster_frames_with_temporal_fitting(features: np.ndarray, times: List[float], window_size: int, eps: float, min_samples: int, debug: bool = False) -> np.ndarray:
    """Cluster frames based on visual similarity and how well they fit within their temporal neighbors."""
    fits = []
    for i in range(len(features)):
        fit_score = evaluate_frame_fit(i, features, window_size)
        fits.append(fit_score)

    combined_features = np.hstack((features, np.array(fits).reshape(-1, 1)))  # Adding fit score as a feature

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(combined_features)

    return clustering.labels_

def cluster_frames_kmeans(features: np.ndarray, num_groups: int, debug: bool = False) -> np.ndarray:
    """Cluster frames using KMeans with a predefined number of groups."""
    debug_print(debug, f"Running KMeans with {num_groups} clusters.")
    kmeans = KMeans(n_clusters=num_groups, n_init=10, verbose=1 if debug else 0).fit(features)

    return kmeans.labels_

def save_frames_flat_by_group(frame_paths: List[str], labels: np.ndarray, output_dir: str, debug: bool = False) -> None:
    """Save frames into the output directory flat, using sequentially named folders, and prefix filenames with the group number."""
    unique_labels = set(labels)

    # Prepare to track group sizes and print meaningful folder names
    label_to_folder = {}
    for idx, label in enumerate(sorted(unique_labels, key=lambda l: -list(labels).count(l))):
        label_to_folder[label] = f"group_{idx + 1}"

    for label in unique_labels:
        # Create group folder inside output directory
        group_dir = os.path.join(output_dir, label_to_folder.get(label, f"unknown_{label}"))
        os.makedirs(group_dir, exist_ok=True)

        group_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        group_frames = [(frame_paths[i], os.stat(frame_paths[i]).st_birthtime) for i in group_indices]
        group_frames_sorted = sorted(group_frames, key=lambda x: x[1])

        # Save frames into their group directory, with group prefix in filenames
        for seq_num, (frame_path, creation_time) in enumerate(group_frames_sorted, 1):
            iso_time = datetime.fromtimestamp(creation_time, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
            group_number = label_to_folder[label].split('_')[1]  # Extract group number
            dest_path = os.path.join(group_dir, f"group_{group_number}_{seq_num}_{iso_time}.png")
            shutil.copy2(frame_path, dest_path)
            debug_print(debug, f"Copied {frame_path} to {dest_path}")

        # Print folder names and file count
        print(f"Created folder: {group_dir} with {len(group_frames_sorted)} files")

    debug_print(debug, f"Saved frames into {len(unique_labels)} groups.")

def process_frames_with_temporal_fit(input_dir: str, output_dir: str, model_name: str, num_groups: Optional[int], eps: float, min_samples: int, window_size: int, limit: Optional[int], checkpoint_interval: int, restore_checkpoint: Optional[str], debug: bool) -> None:
    """Main processing function to extract features, cluster frames with temporal fitting, and save them."""
    frame_paths = load_frame_paths(input_dir, limit=limit, debug=debug)
    total_frames = len(frame_paths)
    print(f"Loaded {total_frames} frames from {input_dir}")

    # Load Hugging Face model and feature extractor
    feature_extractor, model = load_huggingface_model(model_name, debug=debug)

    # Load features from checkpoint if provided
    features = []
    if restore_checkpoint and os.path.exists(restore_checkpoint):
        debug_print(debug, f"Restoring features from checkpoint: {restore_checkpoint}")
        checkpoint_data = np.load(restore_checkpoint)
        features = list(checkpoint_data['features'])
    else:
        # Extract features for all frames
        for i, frame_path in enumerate(frame_paths):
            feature = extract_frame_features(frame_path, feature_extractor, model, debug=debug)
            if feature is not None:
                features.append(feature)

            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(features, "tmp", i + 1, debug=debug)

            update_progress_bar(total_frames, i + 1)
        print()

    # Perform clustering
    if num_groups:
        labels = cluster_frames_kmeans(np.array(features), num_groups, debug=debug)
    else:
        labels = cluster_frames_with_temporal_fitting(np.array(features), list(os.stat(fp).st_birthtime for fp in frame_paths), window_size, eps, min_samples, debug=debug)

    # Output the number of groups found
    unique_labels = set(labels)
    num_groups_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"Number of groups found: {num_groups_found}")

    # Save frames into the group directories
    save_frames_flat_by_group(frame_paths, labels, output_dir, debug=debug)

def update_progress_bar(total_frames: int, current_frame: int) -> None:
    """Display a progress bar for frame processing."""
    sys.stdout.write(f"\rProcessed {current_frame}/{total_frames} frames")
    sys.stdout.flush()

def save_checkpoint(features: List[np.ndarray], output_dir: str, iteration: int, debug: bool = False) -> None:
    """Save progress at a given iteration to a checkpoint file."""
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration}.npz")
    np.savez_compressed(checkpoint_path, features=features)
    debug_print(debug, f"Saved progress at iteration {iteration} to {checkpoint_path}.")

def main() -> None:
    """Main function to parse arguments and call the processing function."""
    parser = argparse.ArgumentParser(description="Sort video frames into groups based on similarity and temporal fitting.")
    parser.add_argument('-i', '--inputDir', required=True, help="Directory containing the video frames.")
    parser.add_argument('-o', '--outputDir', required=True, help="Directory where the sorted frames will be saved.")
    parser.add_argument('-m', '--modelName', default='google/vit-base-patch16-224-in21k', help="Hugging Face model name for feature extraction.")
    parser.add_argument('-n', '--numGroups', type=int, help="Number of groups (clusters) for KMeans.")
    parser.add_argument('-e', '--eps', type=float, default=0.5, help="DBSCAN epsilon (max distance between samples).")
    parser.add_argument('--minSamples', type=int, default=2, help="DBSCAN minimum samples for a cluster.")
    parser.add_argument('--limit', type=int, help="Limit the number of files to process.")
    parser.add_argument('--checkpointInterval', type=int, default=100, help="Interval for saving checkpoints.")
    parser.add_argument('--windowSize', type=int, default=2, help="Window size for temporal fitting.")
    parser.add_argument('--restoreCheckpoint', help="Path to a checkpoint file to restore features and resume processing.")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode to print additional information.")
    args = parser.parse_args()

    process_frames_with_temporal_fit(
        input_dir=args.inputDir,
        output_dir=args.outputDir,
        model_name=args.modelName,
        num_groups=args.numGroups,
        eps=args.eps,
        min_samples=args.minSamples,
        window_size=args.windowSize,
        limit=args.limit,
        checkpoint_interval=args.checkpointInterval,
        restore_checkpoint=args.restoreCheckpoint,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
