import argparse
from datasets import *
import pickle
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.data as data
import torch.nn as nn
from torch.nn import functional as F
import torch
import os
import numpy as np
import sys
import copy
from cifar_models import resnet110
from tqdm import tqdm

# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def obj_cnt(dataloader, args):
    counts = {}
    all_categories = dataloader.dataset.categories
    for i in range(len(all_categories)):
        counts["{0}-{1}".format(i, i)] = 0
        for j in range(i+1, len(all_categories)):
            counts["{0}-{1}".format(i, j)] = 0

    groupings_size = [[] for i in range(len(dataloader.dataset.supercategories_to_names))]
    groupings_dist = [[] for i in range(len(dataloader.dataset.supercategories_to_names))]
    group_mapping = dataloader.dataset.group_mapping
    img_center = np.array([.5, .5])
    instances_size = [[] for i in range(len(all_categories))]

    filepaths = [[] for i in range(len(dataloader.dataset.supercategories_to_names))] # for qualitative examples in analysis step

    with_people_instances = np.zeros(len(all_categories))
    with_people = np.zeros(len(dataloader.dataset.supercategories_to_names))
    not_with_people = np.zeros(len(dataloader.dataset.supercategories_to_names))
    if hasattr(dataloader.dataset, 'people_labels'):
        people = dataloader.dataset.people_labels
    else:
        people = None

    overlap = {}
    overlap_percent = .95

    for i, (data, target) in enumerate(tqdm(dataloader)):
        anns = target[0]
        filepath = target[3]
        categories = list(set([ann['label'] for ann in anns]))
        if people is not None:
            has_people = False
            if len(list(set(people) & set(categories))) > 0:
                has_people = True

        co_added = []
        overlap_added = []
        people_added = []
        sizes_added = []
        for a in range(len(anns)):
            
            bbox = anns[a]['bbox']
            size = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
            if anns[a]['label'] not in sizes_added:
                instances_size[all_categories.index(anns[a]['label'])].append([size, filepath])
                sizes_added.append(anns[a]['label'])
            elif instances_size[all_categories.index(anns[a]['label'])][-1][0] < size:
                instances_size[all_categories.index(anns[a]['label'])][-1][0] = size
            if group_mapping is not None:
                # size of object and distance of object from image center
                group = group_mapping(anns[a]['label'])
                obj_center = np.array([bbox[0] + (bbox[1]/2.), bbox[2] + (bbox[3]/2.)])
                distance = np.linalg.norm(obj_center - img_center)
                groupings_size[group].append(size)
                groupings_dist[group].append(distance)

                # if there's a person in this image or not
                if people is not None:
                    if group not in people_added:
                        if len(filepaths[group]) < 500:
                            filepaths[group].append(filepath)

                        if has_people:
                            with_people[group] += 1
                        else:
                            not_with_people[group] += 1
                        people_added.append(group)

            # instance and cooccurrence counts
            cat_a = dataloader.dataset.categories.index(anns[a]['label'])
            key = '{0}-{1}'.format(cat_a, cat_a)
            if key not in co_added:
                co_added.append(key)
                counts[key] += 1
                if people is not None and has_people:
                    with_people_instances[cat_a] += 1
            for b in range(a+1, len(anns)):
                cat_b = dataloader.dataset.categories.index(anns[b]['label'])
                if cat_a < cat_b:
                    key = "{0}-{1}".format(cat_a, cat_b)
                else:
                    key = "{0}-{1}".format(cat_b, cat_a)
                if 'bbox' in anns[a].keys() and bb_intersection_over_union(anns[a]['bbox'], anns[b]['bbox']) > overlap_percent and anns[a]['label'] != anns[b]['label']:
                    if key not in overlap_added:
                        overlap_added.append(key)
                        if key in overlap.keys():
                            overlap[key] += 1
                        else:
                            overlap[key] = 1
                if key not in co_added:
                    co_added.append(key)
                    counts[key] += 1

    stats = {}
    stats['counts'] = counts
    stats['overlap'] = overlap
    stats['sizes'] = groupings_size
    stats['distances'] = groupings_dist
    stats['with_people'] = with_people
    stats['not_with_people'] = not_with_people
    stats['with_people_instances'] = with_people_instances
    stats['filepaths'] = filepaths
    stats['instances_size'] = instances_size
    pickle.dump(stats, open("results/{}/obj_cnt.pkl".format(args.folder), "wb"))

def obj_siz(dataloader, args):
    obj_cnt(dataloader, args)

def obj_ppl(dataloader, args):
    obj_cnt(dataloader, args)

def obj_scn(dataloader, args):
    info = pickle.load(open('util_files/places_scene_info.pkl', 'rb'))
    idx_to_scene = info['idx_to_scene']
    idx_to_scenegroup = info['idx_to_scenegroup']
    sceneidx_to_scenegroupidx = info['sceneidx_to_scenegroupidx']

    # For overall scene counts
    scenes = np.zeros(len(idx_to_scenegroup))

    # For scene-supercategory cooccurrence counts 
    scene_supercategory = np.zeros((len(idx_to_scenegroup), len(dataloader.dataset.supercategories_to_names)))

    # For scene-instance coocurrence counts
    scene_instance = np.zeros((len(idx_to_scenegroup), len(dataloader.dataset.categories)))

    # For supercategory to scenes to features
    supercat_to_scenes_to_features = {}

    img_center = np.array([.5, .5])
    group_mapping = dataloader.dataset.group_mapping
    categories = dataloader.dataset.categories

    # To get features from entire image
    use_cuda = not args.ngpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    arch = 'resnet18' # There's other architectures available on https://github.com/CSAILVision/places365
    model_file = '%s_places365.pth.tar' % arch
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    scenes_to_features = {}
    for scene in idx_to_scenegroup.keys():
        scenes_to_features[scene] = []
    for i in range(len(dataloader.dataset.supercategories_to_names)):
        supercat_to_scenes_to_features[i] = copy.deepcopy(scenes_to_features)

    # To get features from cropped object instance
    model_file = 'cifar_resnet110.th'
    small_model = resnet110()
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    small_model.load_state_dict(state_dict)
    small_model.to(device)
    small_model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    center_crop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for i, (data, target) in enumerate(tqdm(dataloader)):
        anns = target[0]
        filepath = target[3]
        top_scene = target[4]
        instances = list(set([ann['label'] for ann in anns]))
        for instance in instances:
            index = categories.index(instance)
            for top in top_scene:
                scene_instance[top][index] += 1

        for gr in top_scene:
            scenes[gr] += 1
        if group_mapping is not None:
            added = []
            for instance in instances:
                supercategory = group_mapping(instance)
                for gr in top_scene:
                    key = "{0}-{1}".format(gr, supercategory)
                    if key not in added:
                        added.append(key)
                        scene_supercategory[gr][supercategory] += 1
        if group_mapping is None:
            continue

        ### replacement for just getting small features #####
        data = normalize(data).to(device)
        for ann in anns:
            size = list(data.size())[1:]
            supercat = group_mapping(ann['label'])
            bbox = np.array([ann['bbox'][0]*size[1], ann['bbox'][1]*size[1], ann['bbox'][2]*size[0], ann['bbox'][3]*size[0]]).astype(int)
            instance = data[:, bbox[2]:bbox[3], bbox[0]:bbox[1]]
            if 0 in list(instance.size()):
                continue
            small_data = F.interpolate(instance.unsqueeze(0), size=32, mode='bilinear').to(device)
            small_features = small_model.features(small_data).data.cpu().numpy()
            for gr in top_scene:
                supercat_to_scenes_to_features[supercat][gr].append([small_features, filepath])
        ### replacement for just getting small features #####
    
    info = {}
    info['scenes'] = scenes
    info['scene_supercategory'] = scene_supercategory
    info['scene_instance'] = scene_instance
    info['supercat_to_scenes_to_features'] = supercat_to_scenes_to_features
    pickle.dump(info, open('results/{}/obj_scn.pkl'.format(args.folder), 'wb'))






            
            



