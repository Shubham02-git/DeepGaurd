import os
import json
import pickle
import numpy as np
import heapq
import random
from tqdm import tqdm
from scipy.spatial import KDTree


def load_landmark(file_path):
\
\
\
\
\
\
\
\
\
\
\
       
    if file_path is None:
        return np.zeros((81, 2))
    if os.path.exists(file_path):
        landmark = np.load(file_path)
        return np.float32(landmark)
    else:
        return np.zeros((81, 2))


def get_landmark_dict(dataset_folder):
                                                      
    if os.path.exists('landmark_dict_ff.pkl'):
        with open('landmark_dict_ff.pkl', 'rb') as f:
            return pickle.load(f)
                                                   
    metadata_path = os.path.join(dataset_folder, "FaceForensics++.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
                                                                              
    ff_real_data = metadata['FaceForensics++']['FF-real']
                                                                  
    landmark_dict = {
        frame_path.replace('frames', 'landmarks').replace(".png", ".npy"): load_landmark(
            frame_path.replace('frames', 'landmarks').replace(".png", ".npy")
        )
        for mode, value in ff_real_data.items()
        for video_name, video_info in tqdm(value['c23'].items())
        for frame_path in video_info['frames']
    }
                                          
    with open('landmark_dict_ffall.pkl', 'wb') as f:
        pickle.dump(landmark_dict, f)
    return landmark_dict


def get_nearest_faces_fixed_pair(landmark_info, num_neighbors):
\
\
       
    random.seed(1024)                                           

                                                      
    if os.path.exists('nearest_face_info.pkl'):
        with open('nearest_face_info.pkl', 'rb') as f:
            return pickle.load(f)

    landmarks_array = np.array([lmk.flatten() for lmk in landmark_info.values()])
    landmark_ids = list(landmark_info.keys())

                                                  
    tree = KDTree(landmarks_array)

    nearest_faces = {}
    for idx, this_lmk in tqdm(enumerate(landmarks_array), total=len(landmarks_array)):
                                                                       
        dists, indices = tree.query(this_lmk, k=num_neighbors + 1)
                                                                           
        picked_idx = random.choice(indices[1:])
        nearest_faces[landmark_ids[idx]] = landmark_ids[picked_idx]

                                          
    with open('nearest_face_info.pkl', 'wb') as f:
        pickle.dump(nearest_faces, f)

    return nearest_faces


def get_nearest_faces(landmark_info, num_neighbors):
\
\
       
    random.seed(1024)                                           

                                                      
    if os.path.exists('nearest_face_info.pkl'):
        with open('nearest_face_info.pkl', 'rb') as f:
            return pickle.load(f)

    landmarks_array = np.array([lmk.flatten() for lmk in landmark_info.values()])
    landmark_ids = list(landmark_info.keys())

                                                  
    tree = KDTree(landmarks_array)

    nearest_faces = {}
    for idx, this_lmk in tqdm(enumerate(landmarks_array), total=len(landmarks_array)):
                                                                       
        dists, indices = tree.query(this_lmk, k=num_neighbors + 1)
                                                          
        nearest_faces[landmark_ids[idx]] = [landmark_ids[i] for i in indices[1:]]

                                          
    with open('nearest_face_info.pkl', 'wb') as f:
        pickle.dump(nearest_faces, f)

    return nearest_faces

                                                           
dataset_folder = "/home/zhiyuanyan/disfin/deepfake_benchmark/preprocessing/dataset_json/"
landmark_info = get_landmark_dict(dataset_folder)

                                                         
num_neighbors = 100
nearest_faces_info = get_nearest_faces(landmark_info, num_neighbors)                               
