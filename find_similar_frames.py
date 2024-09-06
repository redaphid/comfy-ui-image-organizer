import os
import argparse
import numpy as np
import cv2
import torch
import shutil
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional

MAX_FILE_SIZE_MB = 10  # Maximum file size in megabytes

def debug_print(debug: bool, message: str) -> None:
    """Print debug information if debug mode is enabled."""
    if debug:
        print(f"DEBUG: {message}")

def load_frame_paths(directory: str, limit: Optional[int] = None, debug: bool = False) -> List[str]:
    """Load frame file paths, sort them by file creation time, and apply the optional limit."""
    frames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

    # Sort by file creation time (use `st_ctime`, or `st_mtime` if creation time is not available)
    frame_paths = sorted(frames, key=lambda x: os.stat(x).st_ctime)

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
    if image is None:
        debug_print(debug, f"Skipping {frame_path}: Unable to load image.")
        return None

    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the feature vector and flatten it for comparison
    feature_vector = outputs.last_hidden_state.mean(dim=1).flatten().numpy()
    return feature_vector

def find_most_similar_output_file(input_feature: np.ndarray, output_features: List[np.ndarray], tolerance: float = 0.9, debug: bool = False) -> Optional[int]:
    """Find the most similar file in the output directory based on cosine similarity, return the index."""
    max_similarity = -1
    most_similar_index = None
    for i, output_feature in enumerate(output_features):
        similarity = cosine_similarity([input_feature], [output_feature])[0][0]
        if similarity > max_similarity and similarity >= tolerance:
            max_similarity = similarity
            most_similar_index = i

    if debug:
        if most_similar_index is not None:
            debug_print(debug, f"Most similar file found at index {most_similar_index} with similarity {max_similarity}.")
        else:
            debug_print(debug, f"No similar file found with similarity >= {tolerance}.")
    return most_similar_index

def process_files(input_dir: str, output_dir: str, feature_extractor: AutoFeatureExtractor, model: AutoModel, debug: bool = False, eps: float = 0.5) -> None:
    """Process files in the input directory and try to match them with files in the output directory."""
    # Load frame paths from input and output directories
    input_paths = load_frame_paths(input_dir, debug=debug)
    output_paths = load_frame_paths(output_dir, debug=debug)

    if not input_paths:
        print("No input frames found. Exiting.")
        return

    if not output_paths:
        print("No output frames found. Exiting.")
        return

    # Extract features for files in the output directory
    output_features = []
    for output_path in output_paths:
        feature = extract_frame_features(output_path, feature_extractor, model, debug=debug)
        if feature is not None:
            output_features.append(feature)

    # Process input files and try to match them to the output directory files
    for input_path in input_paths:
        input_feature = extract_frame_features(input_path, feature_extractor, model, debug=debug)
        if input_feature is not None:
            # Find most similar file in the output directory
            similar_file_idx = find_most_similar_output_file(input_feature, output_features, debug=debug)
            if similar_file_idx is not None:
                # Copy the input file to the corresponding location in the output directory
                output_dest = output_paths[similar_file_idx]
                shutil.copy2(input_path, output_dest)
                debug_print(debug, f"Copied {input_path} to {output_dest}")
            else:
                debug_print(debug, f"No similar file found for {input_path}. Skipping.")


def main() -> None:
    """Main function to process video frames and copy similar ones into the output directory."""
    parser = argparse.ArgumentParser(description="Sort video frames into groups based on similarity.")
    parser.add_argument('-i', '--inputDir', required=True, help="Directory containing the video frames.")
    parser.add_argument('-o', '--outputDir', required=True, help="Directory where the sorted frames will be saved.")
    parser.add_argument('-m', '--modelName', default='google/vit-base-patch16-224-in21k', help="Hugging Face model name for feature extraction.")
    parser.add_argument('-e', '--eps', type=float, default=0.5, help="DBSCAN epsilon (max distance between samples).")
    parser.add_argument('-d', '--debug', action='store_true', help="Enable debug mode to print additional information.")
    args = parser.parse_args()

    # Validate input/output directories
    if not os.path.exists(args.inputDir):
        raise FileNotFoundError(f"Input directory '{args.inputDir}' does not exist.")
    if not os.path.exists(args.outputDir):
        raise FileNotFoundError(f"Output directory '{args.outputDir}' does not exist.")

    # Load Hugging Face model and feature extractor
    feature_extractor, model = load_huggingface_model(args.modelName, debug=args.debug)

    # Process the input files and try to match them to the output directory
    process_files(args.inputDir, args.outputDir, feature_extractor, model, debug=args.debug eps=args.eps)

    print(f"Files have been processed and copied to {args.outputDir} if similar matches were found.")

if __name__ == "__main__":
    main()
