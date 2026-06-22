import re
import time
import random
import requests
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

dataset_name = "Pleumpreeti/Driving-Behavior-Classification"
tree_api_url = f"https://huggingface.co/api/datasets/{dataset_name}/tree/main"
download_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
labels = ["normal", "swerving", "tailgating"]
video_fps = 20
frames_to_check = 32
max_threads = 6
max_tries = 6

def fetch_url(url, params=None):
    error = None
    for attempt in range(max_tries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                wait_time = float(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(wait_time + random.uniform(0, 1))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            error = e
            time.sleep(0.5 * (attempt + 1))
    raise error

def get_folder_items(path):
    url = f"{tree_api_url}/{path}"
    items = []
    while url:
        response = fetch_url(url)
        items.extend(response.json())
        link_header = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        url = match.group(1) if match else None
    return items

def get_frame_numbers(total_frames, wanted_frames):
    if total_frames <= wanted_frames:
        return list(range(total_frames)) + [total_frames - 1] * (wanted_frames - total_frames)
    step = total_frames / wanted_frames
    return [int(i * step) for i in range(wanted_frames)]

def download_image_as_gray(path):
    url = f"{download_url}/{path}"
    response = fetch_url(url)
    image_array = np.frombuffer(response.content, dtype=np.uint8)
    gray_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    return gray_image

root_folders = [item["path"] for item in get_folder_items("") if item["type"] == "directory"]

folder_to_label = {}
for folder in root_folders:
    for label in labels:
        if folder.lower()[:5] == label.lower()[:5]:
            folder_to_label[folder] = label
            break
    else:
        folder_to_label[folder] = folder

video_data = []
for folder in root_folders:
    label = folder_to_label[folder]
    clips = [item for item in get_folder_items(folder) if item["type"] == "directory"]
    for clip in clips:
        video_data.append({
            "label": label,
            "folder_path": clip["path"],
            "clip_name": clip["path"].split("/")[-1]
        })

video_df = pd.DataFrame(video_data)

def get_clip_details(row):
    items = get_folder_items(row["folder_path"])
    frames = sorted(item["path"] for item in items if item["type"] == "file" and item["path"].lower().endswith(".png"))
    sizes = {item["path"]: item.get("size", 0) for item in items if item["type"] == "file"}
    total_size_mb = sum(sizes.get(f, 0) for f in frames) / (1024 * 1024)
    return row.name, frames, total_size_mb

frames_dict = {}
sizes_dict = {}

with ThreadPoolExecutor(max_workers=max_threads) as pool:
    tasks = {pool.submit(get_clip_details, row): i for i, row in video_df.iterrows()}
    for task in as_completed(tasks):
        row_index, frames, total_size_mb = task.result()
        frames_dict[row_index] = frames
        sizes_dict[row_index] = total_size_mb

video_df["frames_list"] = video_df.index.map(frames_dict)
video_df["total_frames"] = video_df["frames_list"].apply(len)
video_df["seconds"] = video_df["total_frames"] / video_fps
video_df["size_in_mb"] = video_df.index.map(sizes_dict)

label_stats = video_df.groupby("label").agg(
    total_clips=("clip_name", "count"),
    total_frames=("total_
