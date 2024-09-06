import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from typing import List
from datetime import datetime
import torch
from torch.hub import load

# RIFE model import (using torch.hub)
def load_rife_model(device: str = 'cuda') -> torch.nn.Module:
    """Load the RIFE model from torch hub for frame interpolation."""
    model = load('hpc2030/rife', 'rife-v4', pretrained=True)
    model.to(device).eval()
    return model

def interpolate_frames(frame1: np.ndarray, frame2: np.ndarray, model: torch.nn.Module, num_interpolations: int = 2, device: str = 'cuda') -> List[np.ndarray]:
    """Use RIFE to interpolate between two frames and return the interpolated frames."""
    frames = []
    input_tensor = torch.from_numpy(np.stack([frame1, frame2])).permute(0, 3, 1, 2).float().div(255.0).to(device)

    # Perform interpolation for the desired number of intermediate frames
    for i in range(1, num_interpolations + 1):
        alpha = i / (num_interpolations + 1)
        with torch.no_grad():
            interpolated_frame = model.infer(input_tensor[0], input_tensor[1], alpha)
        interpolated_frame = (interpolated_frame.permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
        frames.append(interpolated_frame)

    return frames

def load_frame_paths(directory: str) -> List[str]:
    """Load all frame file paths from the directory, sorted by creation time."""
    frames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    frame_paths = sorted(frames, key=lambda x: os.stat(x).st_birthtime)
    return frame_paths

def load_frames(frame_paths: List[str]) -> List[np.ndarray]:
    """Load image frames from the provided file paths."""
    frames = []
    for path in tqdm(frame_paths, desc="Loading frames"):
        image = cv2.imread(path)
        if image is not None:
            frames.append(image)
    return frames

def create_video_with_interpolation(output_path: str, frames: List[np.ndarray], fps: int = 24, num_interpolations: int = 2, device: str = 'cuda') -> None:
    """Create a video from the frames with smart interpolation."""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = load_rife_model(device=device)

    print("Generating video with interpolation...")

    for i in tqdm(range(len(frames) - 1), desc="Interpolating and Writing frames"):
        video_writer.write(frames[i])  # Write the current frame
        interpolated_frames = interpolate_frames(frames[i], frames[i + 1], model, num_interpolations=num_interpolations, device=device)
        for interpolated_frame in interpolated_frames:
            video_writer.write(interpolated_frame)  # Write the interpolated frames

    video_writer.write(frames[-1])  # Write the last frame
    video_writer.release()

    print(f"Video saved to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a video from frames with smart frame interpolation using RIFE.")
    parser.add_argument('-i', '--inputDir', required=True, help="Directory containing the input frames.")
    parser.add_argument('-o', '--outputFile', required=True, help="Path to save the output video file.")
    parser.add_argument('--fps', type=int, default=24, help="Frames per second for the output video.")
    parser.add_argument('--interpolations', type=int, default=2, help="Number of interpolations between each pair of frames.")
    parser.add_argument('--device', default='cuda', help="Device to run the model on (cuda or cpu).")

    args = parser.parse_args()

    # Load frames from the directory
    frame_paths = load_frame_paths(args.inputDir)
    frames = load_frames(frame_paths)

    # Generate video with frame interpolation
    create_video_with_interpolation(args.outputFile, frames, fps=args.fps, num_interpolations=args.interpolations, device=args.device)

if __name__ == "__main__":
    main()
