import os
assert 'measurement' not in os.getcwd() and 'analysis_notebooks' not in os.getcwd(), "Script must be run from home directory"
import argparse
import sys
sys.path.append('.')
from datasets import *
import pickle
import torchvision.transforms as transforms
from torchvision import models
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import cv2
import boto3
import imageio
import botocore
from PIL import Image
from util_files.cifar_models import resnet110
from tqdm import tqdm

# 0 for aws rekognition, 1 for free cv2
FACE_DETECT = 1

# call aws rekognition for face unless info already exists from being called before
def detect_face(filepath, info, client):
    if filepath not in info.keys():
        with open(filepath, 'rb') as image:
            try:
                response = client.detect_faces(Image={'Bytes': image.read()})
            except botocore.exceptions.ClientError as error:
                print(error)
                this_image = Image.open(filepath)
                this_image.thumbnail((500, 500))
                this_image.save('temp.jpg')
                with open('temp.jpg', 'rb') as im:
                    response = client.detect_faces(Image={'Bytes': im.read()})
            info[filepath] = response

    return info

#Note: attribute possible values and names are passed in with args: --attribute_values '2' --attribute_names 'male female'
def att_siz(dataloader, args):
    num_attrs = len(dataloader.dataset.attribute_names)

    sizes = [[] for i in range(num_attrs)]
    tiny_sizes = [[] for i in range(num_attrs)]
    no_faces = [[] for i in range(num_attrs)]
    distances = [[] for i in range(num_attrs)]
    img_center = np.array([.5, .5])

    info = pickle.load(open('util_files/places_scene_info.pkl', 'rb'))
    idx_to_scene = info['idx_to_scene']
    idx_to_scenegroup = info['idx_to_scenegroup']
    sceneidx_to_scenegroupidx = info['sceneidx_to_scenegroupidx']

    if FACE_DETECT == 0:
        client = boto3.client('rekognition')
        if os.path.exists('{}_rekognitioninfo.pkl'.format(args.folder)):
            detect_info = pickle.load(open('{}_rekognitioninfo.pkl'.format(args.folder), 'rb'))
        else:
            detect_info = {}
    elif FACE_DETECT == 1:
        cascPath = "util_files/haarcascade_frontalface_default.xml"
        try:
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
        except AttributeError:
            faceCascade = cv2.CascadeClassifier(cascPath)
        
    for i, (data, target) in enumerate(tqdm(dataloader)):
        attribute = target[1]
        file_path = target[3]
        # Only look at image if there is an attribute to analyze (Note attribute require a bbox around the person or thing to analyze)
        if len(attribute)> 1:
            shape = list(data.size())[1:]
            for index in range(len(attribute[1])):
                bbox = attribute[1][index]
                att = attribute[0][index]
                bbox_adjust = np.array([bbox[0]*shape[1], bbox[1]*shape[1], bbox[2]*shape[0], bbox[3]*shape[0]])
                pixel_size = (bbox_adjust[1]-bbox_adjust[0])*(bbox_adjust[3]-bbox_adjust[2])
                size = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
                person_center = np.array([bbox[0] + (bbox[1]/2.), bbox[2] + (bbox[3]/2.)])
                distance = np.linalg.norm(person_center - img_center)
                pic = data.data.cpu().numpy()

                if FACE_DETECT == 0:
                    detect_info = detect_face(target[3], detect_info, client)
                    faceDetails = detect_info[target[3]]['FaceDetails']
                    if len(detect_info) % 20 == 0:
                        pickle.dump(detect_info, open('{}_rekognitioninfo.pkl'.format(args.folder), 'wb'))

                    yes_face = False
                    for face in faceDetails:
                        if face['Confidence'] > .9:
                            yes_face = True
                elif FACE_DETECT == 1:
                    image = cv2.imread(target[3])
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    try:
                        faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        #minSize=(30, 30),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                    except cv2.error as e:
                        faceCascade = cv2.CascadeClassifier(cascPath)
                        faces = faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        #minSize=(30, 30),
                        flags = cv2.CASCADE_SCALE_IMAGE
                        )
                    yes_face = False
                    if len(faces) > 0:
                        yes_face = True

                # If there's no face detected and the person is too small, attribute cannot be distinguished
                if not yes_face or pixel_size < 1000.:
                    scene_group = target[4]
                    if not yes_face:
                        no_faces[att].append(((size, pixel_size, scene_group), (file_path, i)))
                    elif pixel_size < 1000.:        
                        tiny_sizes[att].append(((size, scene_group), (file_path, i)))
                    continue
                sizes[att].append((size, (file_path, i)))
                distances[att].append((distance, (file_path, i)))

    if FACE_DETECT == 0:
        pickle.dump(detect_info, open('{}_rekognitioninfo.pkl'.format(args.folder), 'wb'))

    stats = {}
    # These 3 sizes dictionaries are all mutually exclusive
    stats['sizes'] = sizes
    stats['tiny_sizes'] = tiny_sizes # tiny sizes that still have a face
    stats['noface_sizes'] = no_faces
    stats['distances'] = distances

    pickle.dump(stats, open("results/{}/att_siz.pkl".format(args.folder), "wb"))

def att_cnt(dataloader, args):
    num_attrs = len(dataloader.dataset.attribute_names)

    counts = [{} for i in range(num_attrs)]

    for i in range(len(dataloader.dataset.categories)):
        for a in range(num_attrs):
            counts[a]["{0}-{1}".format(i, i)]= 0
        for j in range(i+1, len(dataloader.dataset.categories)):
            for a in range(num_attrs):
                counts[a]["{0}-{1}".format(i, j)] = 0
            
    for i, (data, target) in enumerate(tqdm(dataloader)):
        attribute = target[1]
        anns = target[0]
        if len(attribute) > 1:
            categories = list(set([ann['label'] for ann in anns]))
            for a in range(len(categories)):
                cat_a = dataloader.dataset.categories.index(categories[a])
                for att in attribute[0]:
                    counts[att]["{0}-{1}".format(cat_a, cat_a)] += 1
                for b in range(a+1, len(categories)):
                    cat_b = dataloader.dataset.categories.index(categories[b])
                    if cat_a < cat_b:
                        for att in attribute[0]:
                            counts[att]["{0}-{1}".format(cat_a, cat_b)] += 1
                    else:
                        for att in attribute[0]:
                            counts[att]["{0}-{1}".format(cat_b, cat_a)] += 1

    pickle.dump(counts, open("results/{}/att_cnt.pkl".format(args.folder), "wb"))

def att_dis(dataloader, args):
    categories = dataloader.dataset.categories
    attr_names = dataloader.dataset.attribute_names
    num_attrs = len(attr_names)
    distances = [[[] for j in range(num_attrs)] for i in range(len(categories))]
    for i, (data, target) in enumerate(tqdm(dataloader)):
        attr = target[1]
        anns = target[0] 
        file_path = target[3]
        if len(attr) > 1:
            for index in range(len(attr[1])):
                bbox = attr[1][index]
                value = attr[0][index]
                person_center = np.array([bbox[0] + (bbox[1]/2.), bbox[2] + (bbox[3]/2.)])
                person_area = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
                seen_instances = []
                for j, ann in enumerate(anns):
                    ann_center = np.array([ann['bbox'][0] + (ann['bbox'][1]/2.), ann['bbox'][2] + (ann['bbox'][3]/2.)])
                    ann_area = (ann['bbox'][1]-ann['bbox'][0])*(ann['bbox'][3]-ann['bbox'][2])
                    distance = np.linalg.norm(person_center - ann_center)
                    # Finds the person-object pair that has the shortest distance
                    if categories.index(ann['label']) in seen_instances:
                        prev = distances[categories.index(ann['label'])][value][-1]
                        prev_calc_dist = prev[0] / np.sqrt(prev[1]*prev[2])
                        calc_dist = distance / np.sqrt(person_area*ann_area)
                        if calc_dist < prev_calc_dist:
                            distances[categories.index(ann['label'])][value][-1] = (distance, person_area, ann_area, file_path, j, index)
                    else:
                        distances[categories.index(ann['label'])][value].append((distance, person_area, ann_area, file_path, j, index))
                        seen_instances.append(categories.index(ann['label']))

    pickle.dump(distances, open("results/{}/att_dis.pkl".format(args.folder), "wb"))

def att_clu(dataloader, args):
    use_cuda = not args.ngpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Extracts scene features from the entire image
    arch = 'resnet18'
    model_file = 'util_files/%s_places365.pth.tar' % arch
    model = models.__dict__[arch](num_classes=365).to(device)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    scene_classifier = model.fc
    new_classifier = nn.Sequential()
    model.fc = new_classifier

    categories = dataloader.dataset.categories
    attr_names = dataloader.dataset.attribute_names
    num_attrs = len(attr_names)
    scene_features = [[[] for j in range(num_attrs)] for i in range(len(categories))]
    instance_features = [[[]  for j in range(num_attrs)] for i in range(len(categories))]
    scene_filepaths = [[[] for j in range(num_attrs)] for i in range(len(categories))]

    # Extracts features of just the cropped object
    model_file = 'util_files/cifar_resnet110.th'
    small_model = resnet110()
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    small_model.load_state_dict(state_dict)
    small_model.to(device)
    small_model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for i, (data, target) in enumerate(tqdm(dataloader)):
        attr = target[1]
        anns = target[0]
        if len(attr) > 1:
            data.to(device)
            data = normalize(data)
            big_data = F.interpolate(data.unsqueeze(0), size=224, mode='bilinear').to(device)
            this_features = model.forward(big_data)
            logit = scene_classifier.forward(this_features)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            pred = idx[0]

            size = list(data.size())[1:]
            scene_added = []

            for ann in anns:
                index = categories.index(ann['label'])
                bbox = np.array([ann['bbox'][0]*size[1], ann['bbox'][1]*size[1], ann['bbox'][2]*size[0], ann['bbox'][3]*size[0]]).astype(int)
                instance = data[:, bbox[2]:bbox[3], bbox[0]:bbox[1]]
                if 0 in list(instance.size()):
                    continue
                small_data = F.interpolate(instance.unsqueeze(0), size=32, mode='bilinear').to(device)
                this_small_features = small_model.features(small_data)
                for att in attr[0]:
                    if len(scene_features[index][att]) < 500 and index not in scene_added:
                        scene_added.append(index)
                        scene_features[index][att].extend(this_features.data.cpu().numpy())
                        scene_filepaths[index][att].append((target[3], pred))
                    if len(instance_features[index][att]) < 500:
                        instance_features[index][att].extend(this_small_features.data.cpu().numpy())
    stats = {}
    stats['instance'] = instance_features
    stats['scene'] = scene_features
    stats['scene_filepaths'] = scene_filepaths
    pickle.dump(stats, open("results/{}/att_clu.pkl".format(args.folder), "wb"))

def att_scn(dataloader, args):
    num_attrs = len(dataloader.dataset.attribute_names)
    info = pickle.load(open('util_files/places_scene_info.pkl', 'rb'))
    idx_to_scene = info['idx_to_scene']
    idx_to_scenegroup = info['idx_to_scenegroup']
    sceneidx_to_scenegroupidx = info['sceneidx_to_scenegroupidx']

    scenes_per = [[0 for a in range(num_attrs)] for i in range(len(idx_to_scenegroup))]
    for i, (data, target) in enumerate(tqdm(dataloader)):
        attribute = target[1]
        anns = target[0]
        top_scene = target[4]
        if len(attribute) > 1 and top_scene is not None:
            for scene in top_scene:
                for att in attribute[0]:
                    scenes_per[scene][att] += 1

    info_stats = {}
    info_stats['scenes_per'] = scenes_per
    pickle.dump(info_stats, open('results/{}/att_scn.pkl'.format(args.folder), 'wb'))
