import argparse
from datasets import *
import pickle
import torchvision.transforms as transforms
from torchvision import models
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os
import sys
import cv2
import boto3
import imageio
import botocore
from PIL import Image
from cifar_models import resnet110
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
def size_and_distance(dataloader, args):
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
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

    for i, (data, target) in enumerate(tqdm(dataloader)):
        attribute = target[1]
        # Only look at image if there is an attribute to analyze (Note attribute require a bbox around the person or thing to analyze)
        if len(attribute)> 1:
            shape = list(data.size())[1:]
            bbox = attribute[1]
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
                    no_faces[attribute[0]].append((size, pixel_size, scene_group))
                elif pixel_size < 1000.:
                    tiny_sizes[attribute[0]].append((size, scene_group))
                continue

            sizes[attribute[0]].append(size)
            distances[attribute[0]].append(distance)

    if FACE_DETECT == 0:
        pickle.dump(detect_info, open('{}_rekognitioninfo.pkl'.format(args.folder), 'wb'))

    stats = {}
    # These 3 sizes dictionaries are all mutually exclusive
    stats['sizes'] = sizes
    stats['tiny_sizes'] = tiny_sizes # tiny sizes that still have a face
    stats['noface_sizes'] = no_faces
    stats['distances'] = distances

    pickle.dump(stats, open("results/{}/1.pkl".format(args.folder), "wb"))

def count_cooccurrence(dataloader, args):
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
                counts[attribute[0]]["{0}-{1}".format(cat_a, cat_a)] += 1
                for b in range(a+1, len(categories)):
                    cat_b = dataloader.dataset.categories.index(categories[b])
                    if cat_a < cat_b:
                        counts[attribute[0]]["{0}-{1}".format(cat_a, cat_b)] += 1
                    else:
                        counts[attribute[0]]["{0}-{1}".format(cat_b, cat_a)] += 1
    pickle.dump(counts, open("results/{}/2.pkl".format(args.folder), "wb"))
