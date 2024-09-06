import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

def load_frame_paths(directory):
    """Load frame file paths and sort them by timestamp."""
    frames = [f for f in os.listdir(directory) if f.endswith('.png')]
    frame_paths = sorted([os.path.join(directory, f) for f in frames], key=lambda x: int(os.path.basename(x).split('.')[0]))
    return frame_paths

def calculate_timestamp_differences(frame_paths):
    """Calculate the time differences between consecutive frames."""
    timestamps = [int(os.path.basename(f).split('.')[0]) for f in frame_paths]
    return np.diff(timestamps), timestamps

def discover_stride(timestamp_diffs):
    """Use clustering to discover the most common stride."""
    timestamp_diffs = timestamp_diffs.reshape(-1, 1)
    clustering = DBSCAN(eps=500, min_samples=3).fit(timestamp_diffs)  # Adjust eps based on how close time differences should be
    labels = clustering.labels_
    common_stride = np.median([d for d, label in zip(timestamp_diffs.flatten(), labels) if label != -1])
    return common_stride

def group_frames_by_discovered_stride(frame_paths, timestamps, stride):
    """Group frames based on the discovered stride."""
    groups = []
    current_group = []
    prev_timestamp = None

    for frame_path, timestamp in zip(frame_paths, timestamps):
        if prev_timestamp is None or timestamp - prev_timestamp <= stride:
            current_group.append(frame_path)
        else:
            groups.append(current_group)
            current_group = [frame_path]

        prev_timestamp = timestamp

    if current_group:
        groups.append(current_group)

    return groups

def calculate_optical_flow(prev_frame, curr_frame):
    """Calculate optical flow between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

def refine_groups_with_optical_flow(groups, optical_flow_threshold):
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
            flow_magnitude = calculate_optical_flow(prev_frame, curr_frame)

            if flow_magnitude <= optical_flow_threshold:
                refined_group.append(group[i])
            else:
                refined_groups.append(refined_group)
                refined_group = [group[i]]

            prev_frame = curr_frame

        if refined_group:
            refined_groups.append(refined_group)

    return refined_groups

def save_frames_to_directories(groups, output_dir):
    """Save grouped frames into corresponding directories."""
    for i, group in enumerate(groups):
        video_dir = os.path.join(output_dir, f'video_{i+1}')
        os.makedirs(video_dir, exist_ok=True)

        for frame_path in group:
            frame_name = os.path.basename(frame_path)
            dest_path = os.path.join(video_dir, frame_name)
            os.rename(frame_path, dest_path)
            print(f"Moved {frame_path} to {dest_path}")

def sort_frames_by_discovered_stride(input_dir, output_dir, optical_flow_threshold=None):
    """Main function to discover stride, group frames, and optionally refine with optical flow."""
    frame_paths = load_frame_paths(input_dir)
    print(f"Loaded {len(frame_paths)} frames from {input_dir}")

    # Step 1: Calculate timestamp differences
    timestamp_diffs, timestamps = calculate_timestamp_differences(frame_paths)

    # Step 2: Discover stride using timestamp differences
    discovered_stride = discover_stride(timestamp_diffs)
    print(f"Discovered stride: {discovered_stride} milliseconds")

    # Step 3: Group frames based on the discovered stride
    groups = group_frames_by_discovered_stride(frame_paths, timestamps, discovered_stride)
    print(f"Grouped frames into {len(groups)} groups based on discovered stride.")

    # Step 4: Refine groups using optical flow if a threshold is provided
    if optical_flow_threshold is not None:
        groups = refine_groups_with_optical_flow(groups, optical_flow_threshold)
        print(f"Refined frames into {len(groups)} groups based on optical flow.")

    # Step 5: Save the frames into separate directories
    save_frames_to_directories(groups, output_dir)
    print(f"Frames have been sorted and saved to {output_dir}.")
