import skimage.color # pip3 install scikit-image
import matplotlib.pyplot as plt # pip3 install matplotlib
import numpy as np # pip3 install numpy
import cv2 # pip3 install opencv-python
import imagehash # pip3 install imagehash
from PIL import Image
from datetime import datetime
import os
import time
import collections
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Function that creates a list of all subfolders it found in a folder
def find_subfolders(directory):
    subfolders = [Path(f.path) for f in os.scandir(directory) if f.is_dir()]
    for directory in list(subfolders):
        subfolders.extend(find_subfolders(directory))
    return subfolders

# Function that creates a list of hashes for each image found in the folders
def create_imgs_hashes(directory, show_progress_a):
    subfolders = find_subfolders(directory)

    # create list of tuples with files found in directory, format: (path, filename)
    folder_files = [(directory, filename) for filename in os.listdir(directory)]
    if len(subfolders) >= 1:
        for folder in subfolders:
            subfolder_files = [(folder, filename) for filename in os.listdir(folder)]
            folder_files = folder_files + subfolder_files

    # create images hashes
    imgs_hash, delete_index = [], []
    for count, file in enumerate(folder_files):
        try:
            if show_progress_a:
                show_progress(count, folder_files, task='preparing files')
            path = Path(file[0]) / file[1]
            # check if the file is not a folder
            if not os.path.isdir(path):
                try:
                    img_hash = imagehash.average_hash(Image.open(path))
                    obj = {}
                    obj["img_name"] = folder_files[count][1]
                    obj["path"] = path._str
                    obj["hash"] = str(img_hash)
                    imgs_hash.append(obj)
                except:
                    delete_index.append(count)
            else:
                delete_index.append(count)
        except KeyboardInterrupt:
            raise KeyboardInterrupt

    for index in reversed(delete_index):
        del folder_files[index]

    return imgs_hash, folder_files

# Function that displays a progress bar during the search
def show_progress(count, list, task='processing images'):
    if count+1 == len(list):
        print(f"{task}: [{count}/{len(list)}] [{count/len(list):.0%}]", end="\r")
        print(f"{task}: [{count+1}/{len(list)}] [{(count+1)/len(list):.0%}]")          
    else:
        print(f"{task}: [{count}/{len(list)}] [{count/len(list):.0%}]", end="\r")


# Function that generates a dictionary for statistics around the completed process
def generate_stats(directory, start_time, end_time, time_elapsed, total):
    stats = {}
    stats["directory"] = str(Path(directory))

    stats["duration"] = {"start_date": time.strftime("%Y-%m-%d", start_time),
                            "start_time": time.strftime("%H:%M:%S", start_time),
                            "end_date": time.strftime("%Y-%m-%d", end_time),
                            "end_time": time.strftime("%H:%M:%S", end_time),
                            "seconds_elapsed": time_elapsed}

    stats["total_hashes_created"] = total
    return stats

if __name__ == "__main__":    
    # set CLI arguments
    parser = argparse.ArgumentParser(description='Find duplicate or similar images')
    parser.add_argument("-A", "--directory", type=str, help='Directory to search for images.', required=True)
    parser.add_argument("-Z", "--output_directory", type=str, help='(optional) Output directory for the result files. Default is working dir.', required=False, nargs='?', default=None)
    parser.add_argument("-p", "--show_progress", type=bool, help='(optional) Shows the real-time progress.', required=False, nargs='?', choices=[True, False], default=True)

    args = parser.parse_args()

    # create filenames for the output files
    timestamp = str(time.time()).replace(".", "_")
    result_file = "img_hashes_" + timestamp + ".json"
    stats_file = "stats_" + timestamp + ".json"

    if args.output_directory != None:
        dir = args.output_directory
    else:
        dir = os.getcwd()

    if not os.path.exists(dir):
        os.makedirs(dir)

    start_time = time.time()        
    print("Initializing...", end="\r")

    img_hashes, folder_files = create_imgs_hashes(args.directory, args.show_progress)
    total = len(img_hashes)

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 4)
    stats = generate_stats(args.directory, time.localtime(start_time), time.localtime(end_time), time_elapsed, total)

    with open(os.path.join(dir, result_file), "w") as file:
        json.dump(img_hashes, file, ensure_ascii=True, indent=2, sort_keys=True)

    with open(os.path.join(dir, stats_file), "w") as file:
        json.dump(stats, file, ensure_ascii=True, indent=2, sort_keys=True)

    print(f"""\nSaved results into folder {dir} and filenames:\n{result_file}\n{stats_file}""")