import skimage.color # pip3 install scikit-image
import matplotlib.pyplot as plt # pip3 install matplotlib
import numpy as np # pip3 install numpy
import cv2 # pip3 install opencv-python
from datetime import datetime
import os
import time
import collections
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Function that creates a list of matrices for each image found in the folders
def create_imgs_matrix(directory, px_size, show_progress_a):
    subfolders = find_subfolders(directory)

    # create list of tuples with files found in directory, format: (path, filename)
    folder_files = [(directory, filename) for filename in os.listdir(directory)]
    if len(subfolders) >= 1:
        for folder in subfolders:
            subfolder_files = [(folder, filename) for filename in os.listdir(folder)]
            folder_files = folder_files + subfolder_files

    # create images matrix
    imgs_matrix, delete_index = [], []
    for count, file in enumerate(folder_files):
        try:
            if show_progress_a:
                show_progress(count, folder_files, task='preparing files')
            path = Path(file[0]) / file[1]
            # check if the file is not a folder
            if not os.path.isdir(path):
                try:
                    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if type(img) == np.ndarray:
                            img = img[..., 0:3]
                            img = cv2.resize(img, dsize=(px_size, px_size), interpolation=cv2.INTER_CUBIC)
                            
                            if len(img.shape) == 2:
                                img = skimage.color.gray2rgb(img)
                            imgs_matrix.append(img)
                    else:
                        delete_index.append(count)
                except:
                    delete_index.append(count)
            else:
                delete_index.append(count)
        except KeyboardInterrupt:
            raise KeyboardInterrupt

    for index in reversed(delete_index):
        del folder_files[index]

    return imgs_matrix, folder_files

# Function that creates a list of all subfolders it found in a folder
def find_subfolders(directory):
    subfolders = [Path(f.path) for f in os.scandir(directory) if f.is_dir()]
    for directory in list(subfolders):
        subfolders.extend(find_subfolders(directory))
    return subfolders

# Function that calulates the mean squared error (mse) between two image matrices
def mse(image_a, image_b):
    # (Yi - ˆyi)ˆ2
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err

#Function for rotating an image matrix by a 90 degree angle
def rotate_img(image):
    image = np.rot90(image, k=1, axes=(0, 1))
    return image

# Function for checking the quality of compared images, appends the lower quality image to the list
def check_img_quality(image_a, image_b):
    size_img_a = os.stat(image_a).st_size
    size_img_b = os.stat(image_b).st_size
    if size_img_a >= size_img_b:
        return image_a, image_b
    else:
        return image_b, image_a

# Function that displays a progress bar during the search
def show_progress(count, list, task='processing images'):
    if count+1 == len(list):
        print(f"{task}: [{count}/{len(list)}] [{count/len(list):.0%}]", end="\r")
        print(f"{task}: [{count+1}/{len(list)}] [{(count+1)/len(list):.0%}]")          
    else:
        print(f"{task}: [{count}/{len(list)}] [{count/len(list):.0%}]", end="\r")

# Function that plots two compared image files and their mse
def show_img_figs(image_a, image_b, err):
    fig = plt.figure()
    plt.suptitle("MSE: %.2f" % (err))
    # plot first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(image_a, cmap=plt.cm.gray)
    plt.axis("off")
    # plot second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(image_b, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    print("Saving image")
    plt.savefig('my_plot.png')
    plt.close()

# Function for printing filename info of plotted image files
def show_file_info(image_a, image_b):
    image_a = "..." + str(image_a)[-45:]
    image_b = "..." + str(image_b)[-45:]
    print(f"""Duplicate files:\n{image_a} and \n{image_b}\n""")

# Function that searches one directory for duplicate/similar images
def search_one_dir(img_matrices_a, folder_files_a, similarity, show_output=False, show_progress_a=False):

    total = len(img_matrices_a)
    result = {}
    lower_quality = []
    ref = similarity

    # find duplicates/similar images within one folder
    for count_a, image_matrix_a in enumerate(img_matrices_a):
        img_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        while img_id in result.keys():
            img_id = str(int(img_id) + 1)
        if show_progress_a:
            show_progress(count_a, img_matrices_a, task='comparing images')
        for count_b, image_matrix_b in enumerate(img_matrices_a):
            if count_b > count_a and count_a != len(img_matrices_a):
                rotations = 0
                while rotations <= 3:
                    if rotations != 0:
                        image_matrix_b = rotate_img(image_matrix_b)

                    err = mse(image_matrix_a, image_matrix_b)
                    if err < ref:
                        if show_output:
                            show_img_figs(image_matrix_a, image_matrix_b, err)
                            show_file_info(Path(folder_files_a[count_a][0]) / folder_files_a[count_a][1], #0 is the path, 1 is the filename
                                                Path(folder_files_a[count_b][0]) / folder_files_a[count_b][1])
                        if img_id in result.keys():
                            result[img_id]["duplicates"] = result[img_id]["duplicates"] + [str(Path(folder_files_a[count_b][0]) / folder_files_a[count_b][1])]
                        else:
                            result[img_id] = {'filename': str(folder_files_a[count_a][1]),
                                                'location': str(Path(folder_files_a[count_a][0]) / folder_files_a[count_a][1]),
                                                'duplicates': [str(Path(folder_files_a[count_b][0]) / folder_files_a[count_b][1])]}
                        try:                                    
                            high, low = check_img_quality(Path(folder_files_a[count_a][0]) / folder_files_a[count_a][1], Path(folder_files_a[count_b][0]) / folder_files_a[count_b][1])
                            lower_quality.append(str(low))
                        except:
                            pass
                        break
                    else:
                        rotations += 1
                        
    result = collections.OrderedDict(sorted(result.items()))
    lower_quality = list(set(lower_quality))
    
    return result, lower_quality, total

# Function that searches two directories for duplicate/similar images
def search_two_dirs(img_matrices_a, folder_files_a, img_matrices_b, folder_files_b, similarity, show_output=False, show_progress_a=False):

    total = len(img_matrices_a) + len(img_matrices_b)
    result = {}
    lower_quality = []
    ref = similarity

    # find duplicates/similar images between two folders
    for count_a, image_matrix_a in enumerate(img_matrices_a):
        img_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        while img_id in result.keys():
            img_id = str(int(img_id) + 1)
        if show_progress_a:
            show_progress(count_a, img_matrices_a, task='comparing images')
        for count_b, image_matrix_b in enumerate(img_matrices_b):
            rotations = 0
            while rotations <= 3:
                if rotations != 0:
                    image_matrix_b = rotate_img(image_matrix_b)
                    
                err = mse(image_matrix_a, image_matrix_b)
                if err < ref:
                    if show_output:
                        show_img_figs(image_matrix_a, image_matrix_b, err)
                        show_file_info(Path(folder_files_a[count_a][0]) / folder_files_a[count_a][1],
                                            Path(folder_files_b[count_b][0]) / folder_files_b[count_b][1])
                    if img_id in result.keys():
                        result[img_id]["duplicates"] = result[img_id]["duplicates"] + [str(Path(folder_files_b[count_b][0]) / folder_files_b[count_b][1])]
                    else:
                        result[img_id] = {'filename': str(folder_files_a[count_a][1]),
                                            'location': str(Path(folder_files_a[count_a][0]) / folder_files_a[count_a][1]),
                                            'duplicates': [str(Path(folder_files_b[count_b][0]) / folder_files_b[count_b][1])]}
                    try:
                        high, low = check_img_quality(Path(folder_files_a[count_a][0]) / folder_files_a[count_a][1], Path(folder_files_b[count_b][0]) / folder_files_b[count_b][1])
                        lower_quality.append(str(low))
                    except:
                        pass
                    break
                else:
                    rotations += 1

    result = collections.OrderedDict(sorted(result.items()))
    lower_quality = list(set(lower_quality))

    return result, lower_quality, total

# Function that maps the similarity grade to the respective MSE value
def map_similarity(similarity):
    try:
        similarity = float(similarity)
        ref = similarity
    except:      
        if similarity == "low":
            ref = 1000
        # search for exact duplicate images, extremly sensitive, MSE < 0.1
        elif similarity == "high":
            ref = 0.1
        # normal, search for duplicates, recommended, MSE < 200
        else:
            ref = 200
    return ref

# Function that generates a dictionary for statistics around the completed process
def generate_stats(directory_a, directory_b, start_time, end_time, time_elapsed, similarity, total_searched, total_found):
    stats = {}
    stats["directory_1"] = str(Path(directory_a))
    
    if directory_b != None:
        stats["directory_2"] = str(Path(directory_b))
    else:
        stats["directory_2"] = None

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
    parser.add_argument("-A", "--directory_a", type=str, help='Directory to search for images.', required=True)
    parser.add_argument("-B", "--directory_b", type=str, help='(optional) Second directory to search for images.', required=False, nargs='?', default=None)
    parser.add_argument("-Z", "--output_directory", type=str, help='(optional) Output directory for the result files. Default is working dir.', required=False, nargs='?', default=None)
    parser.add_argument("-s", "--similarity", type=type_str_int, help='(optional) Similarity grade.', required=False, nargs='?', default='normal')
    parser.add_argument("-px", "--px_size", type=int, help='(optional) Compression size of images in pixels.', required=False, nargs='?', default=50)
    parser.add_argument("-p", "--show_progress", type=bool, help='(optional) Shows the real-time progress.', required=False, nargs='?', choices=[True, False], default=True)
    parser.add_argument("-o", "--show_output", type=bool, help='(optional) Shows the comapred images in real-time.', required=False, nargs='?', choices=[True, False], default=False)
    
    args = parser.parse_args()

    # create filenames for the output files
    timestamp = str(time.time()).replace(".", "_")
    result_file = "results_" + timestamp + ".json"
    lq_file = "lower_quality_" + timestamp + ".txt"
    stats_file = "stats_" + timestamp + ".json"

    if args.output_directory != None:
        dir = args.output_directory
    else:
        dir = os.getcwd()

    if not os.path.exists(dir):
        os.makedirs(dir)

    start_time = time.time()        
    print("Initializing...", end="\r")

    if args.directory_b == None:
        img_matrices_a, folder_files_a = create_imgs_matrix(args.directory_a, args.px_size, args.show_progress)
        ref = map_similarity(args.similarity)
        result, lower_quality, total = search_one_dir(img_matrices_a, folder_files_a, 
                                                            ref, args.show_output, args.show_progress)
    else:
        img_matrices_a, folder_files_a = create_imgs_matrix(args.directory_a, args.px_size, args.show_progress)
        img_matrices_B, folder_files_b = create_imgs_matrix(args.directory_b, args.px_size, args.show_progress)
        ref = map_similarity(args.similarity)
        result, lower_quality, total = search_two_dirs(img_matrices_a, folder_files_a,
                                                            img_matrices_B, folder_files_b,
                                                            ref, args.show_output, args.show_progress)

    end_time = time.time()
    time_elapsed = np.round(end_time - start_time, 4)
    stats = generate_stats(args.directory_a, args.directory_b, 
        time.localtime(start_time), time.localtime(end_time), time_elapsed, 
        args.similarity, total, len(result))

    with open(os.path.join(dir, result_file), "w") as file:
        json.dump(result, file, ensure_ascii=True, indent=2, sort_keys=True)

    with open(os.path.join(dir, lq_file), "w") as file:
        file.writelines(lower_quality)

    with open(os.path.join(dir, stats_file), "w") as file:
        json.dump(stats, file, ensure_ascii=True, indent=2, sort_keys=True)

    print(f"""\nSaved results into folder {dir} and filenames:\n{result_file} \n{lq_file} \n{stats_file}""")