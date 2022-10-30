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

# Function that maps the similarity grade to the respective MSE value
def map_similarity(similarity):
    try:
        similarity = float(similarity)
        ref = similarity
    except:      
        if similarity == "low":
            ref = 1000
        # search for exact duplicate images, extremly sensitive
        elif similarity == "high":
            ref = 0.1
        # normal, search for duplicates, recommended
        else:
            ref = 10
    return ref

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
                    imgs_hash.append(img_hash)
                except:
                    delete_index.append(count)
            else:
                delete_index.append(count)
        except KeyboardInterrupt:
            raise KeyboardInterrupt

    for index in reversed(delete_index):
        del folder_files[index]

    return imgs_hash, folder_files

# Function that creates a list of all subfolders it found in a folder
def find_subfolders(directory):
    subfolders = [Path(f.path) for f in os.scandir(directory) if f.is_dir()]
    for directory in list(subfolders):
        subfolders.extend(find_subfolders(directory))
    return subfolders

# Function that displays a progress bar during the search
def show_progress(count, list, task='processing images'):
    if count+1 == len(list):
        print(f"{task}: [{count}/{len(list)}] [{count/len(list):.0%}]", end="\r")
        print(f"{task}: [{count+1}/{len(list)}] [{(count+1)/len(list):.0%}]")          
    else:
        print(f"{task}: [{count}/{len(list)}] [{count/len(list):.0%}]", end="\r")

def search_hashes(img_hashes_a, img_hashes_b, folder_files_b, similarity, show_progress_a=False):
    total = len(img_hashes_a) + len(img_hashes_b)
    result = {}

    # find duplicates/similar images between two folders
    for count_a, img_hash_a in enumerate(img_hashes_a):
        img_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        while img_id in result.keys():
            img_id = str(int(img_id) + 1)
        if show_progress_a:
            show_progress(count_a, img_hashes_a, task='comparing images')
        for count_b, img_hash_b in enumerate(img_hashes_b):      
                computational_score = (imagehash.hex_to_hash(img_hash_a["hash"]) - img_hash_b)
                if computational_score < similarity:
                    if img_id in result.keys():
                        result[img_id]["duplicates"] = result[img_id]["duplicates"] + [str(Path(folder_files_b[count_b][0]) / folder_files_b[count_b][1])]
                    else:
                        result[img_id] = {'filename': img_hash_a["img_name"],
                                            'location': img_hash_a["path"],
                                            'duplicates': [str(Path(folder_files_b[count_b][0]) / folder_files_b[count_b][1])]}

    result = collections.OrderedDict(sorted(result.items()))

    return result, total

# Function that generates a dictionary for statistics around the completed process
def generate_stats(directory, start_time, end_time, time_elapsed, similarity, total_searched, total_found):
    stats = {}
    stats["directory"] = str(Path(directory))
    stats["duration"] = {"start_date": time.strftime("%Y-%m-%d", start_time),
                            "start_time": time.strftime("%H:%M:%S", start_time),
                            "end_date": time.strftime("%Y-%m-%d", end_time),
                            "end_time": time.strftime("%H:%M:%S", end_time),
                            "seconds_elapsed": time_elapsed}

    if isinstance(similarity, int):
        stats["similarity_grade"] = "manual"
    else:
        stats["similarity_grade"] = similarity

    stats["similarity_mse"] = map_similarity(similarity)
    stats["total_files_searched"] = total_searched
    stats["total_dupl_sim_found"] = total_found
    return stats


def type_str_int(x):
    try:
        return int(x)
    except:
        return x

if __name__ == "__main__":    
    # set CLI arguments
    parser = argparse.ArgumentParser(description='Find duplicate or similar images')
    parser.add_argument("-A", "--directory", type=str, help='Directory to search for images.', required=True)
    parser.add_argument("-J", "--json-hashes",  type=str, help='JSON with image hashes', required=True)
    parser.add_argument("-Z", "--output_directory", type=str, help='(optional) Output directory for the result files. Default is working dir.', required=False, nargs='?', default=None)
    parser.add_argument("-s", "--similarity", type=type_str_int, help='(optional) Similarity grade.', required=False, nargs='?', default='normal')
    parser.add_argument("-p", "--show_progress", type=bool, help='(optional) Shows the real-time progress.', required=False, nargs='?', choices=[True, False], default=True)
   
    args = parser.parse_args()

    # create filenames for the output files
    timestamp = str(time.time()).replace(".", "_")
    result_file = "results_" + timestamp + ".json"
    stats_file = "stats_" + timestamp + ".json"

    if args.output_directory != None:
        dir = args.output_directory
    else:
        dir = os.getcwd()

    if not os.path.exists(dir):
        os.makedirs(dir)

    start_time = time.time()        
    print("Initializing...", end="\r")

    ref = map_similarity(args.similarity)

    # hash comparing
    with open(args.json_hashes, 'r') as file:
        image_hashes_a = np.array(json.load(file))

    img_hashes_b, folder_files_b =  create_imgs_hashes(args.directory, args.show_progress)
    result, total = search_hashes(image_hashes_a, img_hashes_b, folder_files_b, ref, args.show_progress)

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 4)
    stats = generate_stats(args.directory, time.localtime(start_time), time.localtime(end_time), time_elapsed, 
        args.similarity, total, len(result))

    with open(os.path.join(dir, result_file), "w") as file:
        json.dump(result, file, ensure_ascii=True, indent=2, sort_keys=True)

    with open(os.path.join(dir, stats_file), "w") as file:
        json.dump(stats, file, ensure_ascii=True, indent=2, sort_keys=True)

    print(f"""\nSaved results into folder {dir} and filenames:\n{result_file} \n{stats_file}""")