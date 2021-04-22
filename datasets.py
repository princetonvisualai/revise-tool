import argparse
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image
import csv
import os
import pickle
import torch
import xml.etree.ElementTree as ET
import re
from lxml import etree
import pandas
from scipy.io import loadmat
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch
import spacy
from scipy.special import softmax 
import numpy as np
from collections import OrderedDict 
nlp = spacy.load("en_core_web_lg")
import json

def collate_fn(batch):
    return batch[0]

class NoneDict(dict):
    def __getitem__(self, key):
        return dict.get(self, key)

def setup_scenemapping(dataset, name):
    info = pickle.load(open('util_files/places_scene_info.pkl', 'rb'))
    idx_to_scene = info['idx_to_scene']
    idx_to_scenegroup = info['idx_to_scenegroup']
    sceneidx_to_scenegroupidx = info['sceneidx_to_scenegroupidx']

    dataloader = data.DataLoader(dataset=dataset, 
                  num_workers=0,
                  batch_size=1,
                  collate_fn=collate_fn,
                  shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    arch = 'resnet18' # There's other architectures available on https://github.com/CSAILVision/places365
    model_file = 'util_files/%s_places365.pth.tar' % arch
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    center_crop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scene_mapping = {}

    for i, (img, target) in enumerate(dataloader):
        filepath = target[3]
        input_img = Variable(center_crop(img).unsqueeze(0)).to(device)
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        top_scene = sceneidx_to_scenegroupidx[int(idx[0].data.cpu().numpy())]
        scene_mapping[filepath] = top_scene
        if i % 100000 == 0:
            pickle.dump(scene_mapping, open('dataloader_files/{0}_scene_mapping_{1}.pkl'.format(name, i), 'wb'))

    dataset.scene_mapping = scene_mapping
    pickle.dump(scene_mapping, open('dataloader_files/{}_scene_mapping.pkl'.format(name), 'wb'))

def read_xml_content(xml_file):
    parser = etree.XMLParser(recover=True)
    #tree = ET.parse(xml_file)
    tree = ET.parse(xml_file, parser=parser)
    root = tree.getroot()

    list_with_all_boxes = []
    filename = root.find('filename').text
    width, height = float(root.find('size').find('width').text), float(root.find('size').find('height').text) # x is width

    for boxes in root.iter('object'):

        instance = boxes.find('name').text.strip().lower()
        instance = ' '.join(instance.split())
        instance = instance.replace('occluded', '').replace('crop', '').strip()

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        this_box = {'bbox': [xmin / width, xmax / width, ymin / height, ymax / height], 'label': instance}
        list_with_all_boxes.append(this_box)

    return list_with_all_boxes

DEFAULT_GROUPINGS_TO_NAMES = {
    0: 'person',
    1: 'vehicle',
    2: 'outdoor',
    3: 'animal',
    4: 'accessory',
    5: 'sports',
    6: 'kitchen',
    7: 'food',
    8: 'furniture',
    9: 'electronic',
    10: 'appliance',
    11: 'indoor'
}

def group_mapping_creator(labels_to_names, supercategories_to_names=DEFAULT_GROUPINGS_TO_NAMES, 
                          override_map = None):
    '''
    inputs:
    labels_to_names: dict mapping "label" to "human readable string"
    supercategories_to_names: dict mapping "supercat" to "human readable string"
    override_map: dict mapping: "human-readable label" to "human-readable supercat"
                        for manual overriding of mapping of certain labels
           
    output:
    prints out human-readable label to human readable supercat mapping
    
    returns:
    function that takes an input label, and returns a supercat
    '''
    assert (labels_to_names is not None and supercategories_to_names is not None)
    # precompute the spacy tokens for each of the supercategories
    supercat_list = list(supercategories_to_names.keys())
    nlp_supercat = []
    for supercat in supercat_list:
        nlp_supercat.append(nlp(supercategories_to_names.get(supercat)))
    ######################################################################  
    # function that calculates the "closest" supercategory to input word
    # and returns said supercategory and the associated softmax confidence score
    def dist_calculator(word):
        assert(word is not None)
        # score array holding distance btwn word and each supercat, for softmax calculation
        score_arr = []
        for supercat_token in nlp_supercat:
            cur_score = supercat_token.similarity(nlp(word))
            score_arr.append(cur_score)
        score_arr = softmax(score_arr)
        return supercat_list[np.argmax(score_arr)], np.max(score_arr)
    ######################################################################    
    # this represents the result map from label to supercat
    result_label_to_group_map = OrderedDict()
    # (hr_label) is the human-readable value corresponding to the key (label)
    for label, hr_label in labels_to_names.items():
        if override_map and hr_label in override_map:
            # skip if user is overriding with override_map
            continue
        # retrieve tuple result from dist_calculator
        supercat_match = dist_calculator(hr_label)
        result_label_to_group_map[label] = supercat_match
        
    # sort from least to most confident based on softmax scores
    result_label_to_group_map = OrderedDict(sorted(result_label_to_group_map.items(), key=lambda item: item[1][1]))
    
    # remove the softmax scores from the dictionary
    for k, v in result_label_to_group_map.items():
        result_label_to_group_map[k] = v[0]
    
    # add the harded-coded override_map values if provided:
    if override_map:
        # initialize non-human-readable form of override_map
        override_nonhuman_readable = {}
        # invert the labels_to_names 
        names_to_labels = dict((v, k) for k, v in labels_to_names.items())
        # invert the supercategories_to_names
        names_to_supercategories = dict((v, k) for k, v in supercategories_to_names.items())
        # convert human-readable labels and supercats 
        for k,v in override_map.items():
            label = names_to_labels.get(k)
            supercat = names_to_supercategories.get(v)
            assert label in labels_to_names and supercat in supercategories_to_names, "override string not valid"
            override_nonhuman_readable[label] = supercat
        # add overriden dict to result dict 
        result_label_to_group_map.update(override_nonhuman_readable)

    # print mapping in human-readable form so user can adjust if necessary
    print("Here are 20 of the least confident labels to supercategory mappings, ranked in increasing confidence.\nChange as necessary using override_map)")
    print("-------------------------------")
    for entry in list(result_label_to_group_map.items())[:20]:
        print("{0}: {1}".format(labels_to_names.get(entry[0]), supercategories_to_names.get(entry[1])))

    # return function of mapping from label-> supercat
    return lambda label: result_label_to_group_map.get(label)

class TemplateDataset(data.Dataset):
    
    def __init__(self, transform):
        self.transform = transform
        
        # Where the images are located (doesn't need to exist, but can be helpful for other functions)
        self.img_folder = ''

        # List of all of the image ids
        # Note: This is some representation of the image, can be integer or name of image
        self.image_ids = [] 

        # Maps label to the human-readable name
        self.labels_to_names = {}

        # List of all the labels
        self.categories = []

        # Names of attribute values to analyze
        self.attribute_names = ["Female", "Male"]

        # Maps from filepath to scenes
        # Can be set up by running AlexNet Places365 model by running the following command:
        # self.scene_mapping = setup_scenemapping(self, '[name of dataset]')
        self.scene_mapping = NoneDict()
        
        # default to DEFAULT_GROUPINGS_TO_NAMES
        self.supercategories_to_names = DEFAULT_GROUPINGS_TO_NAMES

        #Note: Any of the 'optional' attributes may be necessary depending on analysis and metrics, check before not filling in


        # Maps each label to number of supercategory group, (optional)
        self.group_mapping = group_mapping_creator(self.labels_to_names, self.supercategories_to_names)

        # Labels, that are entries from self.categories, that correspond to people (optional)
        self.people_labels = []

        # Number of images from dataset that are female at index 0 and male at index 1 (optional, doesn't need to exist)
        self.num_attribute_images = [0, 0]
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_path = self.img_folder + '/' + image_id + '.jpg'
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[:-4]

        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)

        #Note: bbox digits should be: x, y, width, height. all numbers are scaled to be between 0 and 1
        person_bbox = None # optional
        #Note: gender should be in a list because attribute_based assumes each image can have more than 1 value
        gender = [None] # optional, we have used 0 for male and 1 for female when these labels exist (yes, this order is reversed from self.num_gender_images above)
        gender_info = [gender, [person_bbox]] # optional Note bbox should also be a list for the same reason mentioned above for gender

        country = None # optional

        #Note: This is a map of present labels: image_anns = [{‘label’: ‘tennis_ball’}, {‘label’: ‘dog’}] where the image has a tennis ball and a dog
        image_anns = None

        scene_group = self.scene_mapping[file_path] # optional

        #optional. lat_lng is a dictionary with 2 keys: 'lat' and 'lng'
        # whose values are type doubles. 
        lat_lng = None 

        #Note: Gender info should not be in an array since gender_info is already array
        anns = [image_anns, [gender_info], [country, lat_lng], file_path, scene_group]

        return image, anns

class OpenImagesDataset(data.Dataset):
    
    def __init__(self, transform):
        self.transform = transform
        
        self.img_folder = 'Data/OpenImages/'
        with open('Data/OpenImages/train-images-boxable-with-rotation.csv', newline='') as csvfile:
            data = list(csv.reader(csvfile))[1:]

            # first line for subset of dataset, second line for full
            # self.image_ids = [chunk[0] for chunk in data if (chunk[0][0] == '0')]
            self.image_ids = [chunk[0] for chunk in data]
        
        self.setup_anns()
        names = list(csv.reader(open('Data/OpenImages/class-descriptions-boxable.csv', newline='')))
        self.labels_to_names = {name[0]: name[1] for name in names}
        self.categories = list(self.labels_to_names.keys())
        self.attribute_names = ["Female", "Male"]

        self.scene_mapping = NoneDict()
        if os.path.exists('dataloader_files/openimages_scene_mapping.pkl'):
            self.scene_mapping = pickle.load(open('dataloader_files/openimages_scene_mapping.pkl', 'rb'))
        else:
            setup_scenemapping(self, 'openimages')

        self.group_mapping = None
        self.people_labels = ['/m/01bl7v', '/m/04yx4', '/m/03bt1vf', '/m/05r655'] # keys in self.categories
        self.people_labels = self.people_labels + ['/m/014sv8', '/m/0283dt1', '/m/02p0tk3', '/m/031n1', '/m/035r7c', '/m/039xj_', '/m/03q69', '/m/04hgtk', '/m/0dzct', '/m/0dzf4', '/m/0k0pj', '/m/0k65p', '/m/015h_t']
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]

        file_path = os.path.join(self.img_folder, 'train_' + image_id[0], image_id) + '.jpg'

        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[:-4]
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)

        anns = self.anns[image_id]
        anns.append(file_path)
        anns.append(self.scene_mapping[file_path])

        return image, anns

    def setup_anns(self):
        if os.path.exists('dataloader_files/openimage_anns.pkl'):
            info = pickle.load(open('dataloader_files/openimage_anns.pkl', 'rb'))
            self.anns = info['anns']
            self.num_attribute_images = info['num_gender']
        else:
            with open('Data/OpenImages/train-annotations-bbox.csv', newline='') as csvfile:
                data = list(csv.reader(csvfile))[1:]
                # bbox is normalized to be between 0 and 1 and of the form [xmin, xmax, ymin, ymax]
                # so to retrieve piece, do image[bbox[2]:bbox[3], bbox[0]:bbox[1]]

                self.anns = {}
                for chunk in data:
                    new_ann = {'bbox': [float(chunk[4]), float(chunk[5]), float(chunk[6]), float(chunk[7])], 'label': chunk[2]}
                    if chunk[0] in self.anns.keys():
                        self.anns[chunk[0]].append(new_ann)
                    else:
                        self.anns[chunk[0]] = [new_ann]

            self.num_attribute_images = [0, 0]
            men = ['/m/01bl7v', '/m/04yx4']
            women = ['/m/03bt1vf', '/m/05r655']
            for key in self.anns.keys():
                biggest_person = 0
                biggest_bbox = 0
                m_presence = 0
                w_presence = 0
                for i in range(len(self.anns[key])):
                    if self.anns[key][i]['label'] in men:
                        m_presence += 1
                        this_bbox = self.anns[key][i]['bbox']
                        this_person = (this_bbox[1]-this_bbox[0])*(this_bbox[3]-this_bbox[2])
                        if this_person > biggest_person:
                            biggest_person = this_person
                            biggest_bbox = this_bbox
                    elif self.anns[key][i]['label'] in women:
                        w_presence += 1
                        this_bbox = self.anns[key][i]['bbox']
                        this_person = (this_bbox[1]-this_bbox[0])*(this_bbox[3]-this_bbox[2])
                        if this_person > biggest_person:
                            biggest_person = this_person
                            biggest_bbox = this_bbox

                if m_presence > 0 and w_presence == 0:
                    self.anns[key] = [self.anns[key], [[1], [biggest_bbox]], [0]]
                    self.num_attribute_images[1] += 1
                elif w_presence > 0 and m_presence == 0:
                    self.anns[key] = [self.anns[key], [[0], [biggest_bbox]], [0]]
                    self.num_attribute_images[0] += 1
                else:
                    self.anns[key] = [self.anns[key], [0], [0]]
            info = {}
            info['anns'] = self.anns
            info['num_gender'] = self.num_attribute_images
            pickle.dump(info, open('dataloader_files/openimage_anns.pkl', 'wb'))

class CoCoDataset(data.Dataset):

    def __init__(self, transform):
        self.transform = transform
        
        self.supercategories_to_names = DEFAULT_GROUPINGS_TO_NAMES
        self.img_folder = 'Data/Coco/2014data/train2014'
        self.coco = COCO('Data/Coco/2014data/annotations/instances_train2014.json')
        gender_data = pickle.load(open('Data/Coco/2014data/bias_splits/train.data', 'rb'))
        self.attribute_data = {int(chunk['img'][15:27]): chunk['annotation'][0] for chunk in gender_data}

        ids = list(self.coco.anns.keys())
        self.image_ids = list(set([self.coco.anns[this_id]['image_id'] for this_id in ids]))

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.labels_to_names = {}
        for cat in cats:
            self.labels_to_names[cat['id']] = cat['name']

        self.categories = list(self.labels_to_names.keys())
        self.attribute_names = ["Female", "Male"]
        self.scene_mapping = NoneDict()
        if os.path.exists('dataloader_files/coco_scene_mapping.pkl'):
            self.scene_mapping = pickle.load(open('dataloader_files/coco_scene_mapping.pkl', 'rb'))
        elif os.path.exists('results/coco_example/coco_scene_mapping.pkl'):
            self.scene_mapping = pickle.load(open('results/coco_example/coco_scene_mapping.pkl', 'rb'))
        else:
            setup_scenemapping(self, 'coco')

        def mapping(ind):
            if ind == 1:
                return 0
            elif ind < 10:
                return 1
            elif ind < 16:
                return 2
            elif ind < 26:
                return 3
            elif ind < 34:
                return 4
            elif ind < 44:
                return 5
            elif ind < 52:
                return 6
            elif ind < 62:
                return 7
            elif ind < 72:
                return 8
            elif ind < 78:
                return 9
            elif ind < 84:
                return 10
            else:
                return 11
        self.group_mapping = mapping # takes in label name, so from self.categories

        self.people_labels = [1] # instances of self.categories
        self.num_attribute_images = [6642, 16324]
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        path = self.coco.loadImgs(image_id)[0]["file_name"]

        file_path = os.path.join(self.img_folder, path)
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)
    
    # helper function if using step 0.5 in README to initialize
    # folder_path so from_path_prerun() can access correct
    # data location
    def init_folder_path(self, folder_path):
        self.folder_path = folder_path
    
    # only if using step 0.5 in README, copy of from_path() except
    # with filename modification to access data path 
    def from_path_prerun(self, file_path):
        image_id = int(os.path.basename(file_path)[-16:-4])
        # need for scene map since the dict uses 
        # original file name as key
        original_file_path = file_path
        # change file_path to one with right folder_path
        _, tail = os.path.split(file_path)
        file_path = self.folder_path + tail

        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)
        image_size = list(image.size())[1:]

        annIds = self.coco.getAnnIds(imgIds=image_id);
        coco_anns = self.coco.loadAnns(annIds) # coco is [x, y, width, height]
        formatted_anns = []
        biggest_person = 0
        biggest_bbox = 0
        for ann in coco_anns:
            bbox = ann['bbox']
            bbox = [bbox[0] / image_size[1], (bbox[0]+bbox[2]) / image_size[1], bbox[1] / image_size[0], (bbox[1]+bbox[3]) / image_size[0]]
            new_ann = {'bbox': bbox, 'label': ann['category_id']}
            formatted_anns.append(new_ann)

            if ann['category_id'] == 1:
                area = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
                if area > biggest_person:
                    biggest_person = area
                    biggest_bbox = bbox

        scene = self.scene_mapping.get(original_file_path, None)
        if biggest_bbox != 0 and image_id in self.attribute_data.keys():
            anns = [formatted_anns, [[self.attribute_data[image_id]], [biggest_bbox]], [0], file_path, scene]
        else:
            anns = [formatted_anns, [0], [0], file_path, scene]
        return image, anns        

    def from_path(self, file_path):
        image_id = int(os.path.basename(file_path)[-16:-4])

        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)
        image_size = list(image.size())[1:]

        annIds = self.coco.getAnnIds(imgIds=image_id);
        coco_anns = self.coco.loadAnns(annIds) # coco is [x, y, width, height]
        formatted_anns = []
        biggest_person = 0
        biggest_bbox = 0
        for ann in coco_anns:
            bbox = ann['bbox']
            bbox = [bbox[0] / image_size[1], (bbox[0]+bbox[2]) / image_size[1], bbox[1] / image_size[0], (bbox[1]+bbox[3]) / image_size[0]]
            new_ann = {'bbox': bbox, 'label': ann['category_id']}
            formatted_anns.append(new_ann)

            if ann['category_id'] == 1:
                area = (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
                if area > biggest_person:
                    biggest_person = area
                    biggest_bbox = bbox

        scene = self.scene_mapping.get(file_path, None)
        if biggest_bbox != 0 and image_id in self.attribute_data.keys():
            anns = [formatted_anns, [[self.attribute_data[image_id]], [biggest_bbox]], [0], file_path, scene]
        else:
            anns = [formatted_anns, [0], [0], file_path, scene]

        return image, anns
    
class CoCoDatasetNoImages(data.Dataset):

    def __init__(self, transform):
        self.supercategories_to_names = DEFAULT_GROUPINGS_TO_NAMES
        self.coco = COCO('Data/Coco/2014data/annotations/instances_train2014.json')
        
        ids = list(self.coco.anns.keys())
        self.image_ids = list(set([self.coco.anns[this_id]['image_id'] for this_id in ids]))

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.labels_to_names = {}
        for cat in cats:
            self.labels_to_names[cat['id']] = cat['name']

        self.categories = list(self.labels_to_names.keys())

        def mapping(ind):
            if ind == 1:
                return 0
            elif ind < 10:
                return 1
            elif ind < 16:
                return 2
            elif ind < 26:
                return 3
            elif ind < 34:
                return 4
            elif ind < 44:
                return 5
            elif ind < 52:
                return 6
            elif ind < 62:
                return 7
            elif ind < 72:
                return 8
            elif ind < 78:
                return 9
            elif ind < 84:
                return 10
            else:
                return 11
        self.group_mapping = mapping # takes in label name, so from self.categories

        self.people_labels = [1] # instances of self.categories
        self.num_attribute_images = [6642, 16324]

class SUNDataset(data.Dataset):

    def __init__(self, transform):
        self.transform = transform
        
        self.img_folder = 'Data/SUN/SUN2012pascalformat/JPEGImages'
        self.annotations_folder = 'Data/SUN/SUN2012pascalformat/Annotations'
        with open('Data/SUN/SUN2012pascalformat/ImageSets/Main/train.txt') as f:
            content = f.readlines()
        self.image_ids = [x.strip() for x in content] 
        with open('Data/SUN/SUN2012pascalformat/ImageSets/Main/test.txt') as f:
            content = f.readlines()
        self.image_ids = self.image_ids + [x.strip() for x in content]

        class KeyDict(dict):
            def __missing__(self, key):
                return key
        self.labels_to_names = KeyDict()

        if os.path.exists('dataloader_files/sun_categories.pkl'):
            self.categories = pickle.load(open('dataloader_files/sun_categories.pkl', 'rb'))
        else:
            categories = [" ".join(re.split(r" {2,}", x.strip())[1:-2]) for x in open('Data/SUN/SUN2012pascalformat/report.txt').readlines()][1:]
            self.categories = list(set([chunk.replace("occluded", "").replace("crop", "").strip() for chunk in categories]))
            pickle.dump(self.categories, open('dataloader_files/sun_categories.pkl', 'wb'))
        
        self.scene_mapping = NoneDict()
        if os.path.exists('dataloader_files/sun_scene_mapping.pkl'):
            self.scene_mapping = pickle.load(open('dataloader_files/sun_scene_mapping.pkl', 'rb'))
        else:
            setup_scenemapping(self, 'sun')


        self.group_mapping = None
        self.people_labels = []
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_path = self.img_folder + '/' + image_id + '.jpg'
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[:-4]

        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)
        image_size = list(image.size())[1:]

        formatted_anns = read_xml_content(self.annotations_folder + '/' + image_id + '.xml')
        anns = [formatted_anns, [0], [0], file_path, self.scene_mapping[file_path]]

        return image, anns

class ImagenetDataset(data.Dataset):

    def __init__(self, transform):
        self.transform = transform
        
        self.supercategories_to_names = DEFAULT_GROUPINGS_TO_NAMES
        self.img_folder = 'Data/ImageNet/ILSVRC_2014_Images/ILSVRC2014_DET_train'
        self.annotations_folder = 'Data/ImageNet/ILSVRC_2014_Annotations/ILSVRC2014_DET_bbox_train'
        self.image_ids = [str(num).zfill(8) for num in range(1, 60659)]

        meta = loadmat('Data/ImageNet/ILSVRC_2014_Devkit/ILSVRC2014_devkit/data/meta_det.mat')['synsets'][0]
        self.labels_to_names = {chunk[1][0]: chunk[2][0] for chunk in meta if chunk[0][0] < 201}

        self.categories = list(self.labels_to_names.keys())
        
        self.scene_mapping = NoneDict()
        if os.path.exists('dataloader_files/imagenet_scene_mapping.pkl'):
            self.scene_mapping = pickle.load(open('dataloader_files/imagenet_scene_mapping.pkl', 'rb'))
        else:
            setup_scenemapping(self, 'imagenet')


        self.group_mapping = group_mapping_creator(self.labels_to_names, self.supercategories_to_names)
        self.people_labels = ['n00007846'] # person, index 124
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_path = self.img_folder + '/ILSVRC2014_train_' + image_id[:4] + '/ILSVRC2014_train_' + image_id + '.JPEG'
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)[17:-5]

        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)
        image_size = list(image.size())[1:]

        formatted_anns = read_xml_content(self.annotations_folder + '/ILSVRC2014_train_' + image_id[:4] + '/ILSVRC2014_train_' + image_id + '.xml')
        anns = [formatted_anns, [0], [0], file_path, self.scene_mapping[file_path]]

        return image, anns

class YfccPlacesDataset(data.Dataset):
    
    def __init__(self, transform, metric='obj_cnt'):
        self.transform = transform
        
        self.img_folder = 'Data/YFCC100m/data/images'

        self.mapping = pickle.load(open('Data/YFCC100m/yfcc_mappings.pkl', 'rb')) #7.6GB
        self.inv_mapping = {v: k for k, v in self.mapping.items()}

        df = pandas.read_csv('Data/YFCC100m/placemeta_train.csv') #1.6GB
        self.with_country = df.loc[df['type'] == 'Country']
        if os.path.exists('dataloader_files/yfcc_anns.pkl'): # 3.2GB
            info = pickle.load(open('dataloader_files/yfcc_anns.pkl', 'rb')) #shuffled
            self.image_ids = info['image_ids']
            self.annotations = info['annotations']
            self.alllang_ids = info['alllang']
            self.all_ids = info['all']
        else:
            self.annotations = {} # image id: annotations
            with open('Data/YFCC100m/tag-train', 'r') as f: #1.5 GB
                content = f.readlines()
                for entry in content:
                    pieces = entry.split()
                    self.annotations[pieces[1]] = [{'label': label} for label in pieces[3].split(',')]

            info = {}
            self.image_ids = list(set(self.with_country['photoid'].values))
            self.image_ids = [str(num) for num in self.image_ids]
            info['all'] = list(set(self.image_ids) & set(self.mapping.keys()))
            self.image_ids = [an_id for an_id in self.image_ids if (an_id in self.mapping.keys() and an_id in self.annotations.keys())]
            info['annotations'] = self.annotations
            info['image_ids'] = self.image_ids
            info['alllang'] = list(pickle.load(open('Data/YFCC100m/tags/YFCC100M/alllang_ids.pkl', 'rb')).keys()) #147M
            random.shuffle(info['all'])
            random.shuffle(info['image_ids'])
            random.shuffle(info['alllang'])
            pickle.dump(info, open('dataloader_files/yfcc_anns.pkl', 'wb'))


        class KeyDict(dict):
            def __missing__(self, key):
                return key
        self.labels_to_names = KeyDict()
        with open('Data/YFCC100m/tags.txt', 'r') as f: # 66K
            content = f.readlines()
        self.categories = [x.strip() for x in content]

        #self.scene_mapping = NoneDict()
        #if os.path.exists('dataloader_files/yfcc_scene_mapping.pkl'):
        #    self.scene_mapping = pickle.load(open('dataloader_files/yfcc_scene_mapping.pkl', 'rb'))
        #else:
        #    setup_scenemapping(self, 'yfcc')

        self.group_mapping = None
        self.people_labels = []

        if metric in ['obj_cnt', 'geo_tag']:
            self.version = 'intersect' # has tags from cleaned English, and geolocation
        elif metric in ['geo_lng']:
            self.version = 'alllang' # has tags in any language, and geolocation
        elif metric in ['geo_ctr']:
            self.version = 'all' # has geolocation
        else:
            raise Exception("Metric can't be run on this dataset")        

        if self.version == 'alllang':
            self.image_ids = self.alllang_ids
            self.mapping_id_to_trainline = pickle.load(open('Data/YFCC100m/tags/YFCC100M/alllang_ids.pkl', 'rb'))
            with open('Data/YFCC100m/tags/YFCC100M/train') as my_file: #19GB
                self.tags = my_file.readlines()
        elif self.version == 'intersect':
            pass
        elif self.version == 'all':
            self.image_ids = self.all_ids

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        m5hash = self.mapping[image_id]
        file_path = self.img_folder + '/' + m5hash[:3] + '/' + m5hash[3:6] + '/' + m5hash + '.jpg'
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        m5hash = os.path.basename(file_path)[:-4]
        image_id = self.inv_mapping[m5hash]

        if self.version == 'alllang' or self.version == 'intersect':
            try:
                if not os.path.exists(file_path):
                    loc = m5hash[:3] + '/' + m5hash[3:6] + '/' + m5hash + '.jpg'
                    print("downloading")
                    os.system('aws s3 cp s3://multimedia-commons/data/images/{0} {1}'.format(loc, file_path))
                    if not os.path.exists(file_path):
                        image = None
                    else:
                        image = Image.open(file_path).convert("RGB")
                        image = self.transform(image)
                else:
                    image = Image.open(file_path).convert("RGB")
                    image = self.transform(image)
            except OSError as e:
                print("OS Error: {}".format(e))
                image = None

        country = self.with_country.loc[self.with_country['photoid'] == int(image_id)]['placename'].values
        if len(country) > 1:
            country = country[list(country).index('United+Kingdom')]
        else:
            country = country[0]

        if self.version == 'alllang':
            trainline = self.mapping_id_to_trainline[image_id]
            this_tags = self.tags[trainline-1].strip() 
            this_tags = re.findall('__label__([^\s]*)\s', this_tags)
            formatted_anns = [{'label': tag} for tag in this_tags]
            anns = [formatted_anns, [0], [country], file_path, None]
        elif self.version == 'intersect':
            anns = [self.annotations[image_id], [0], [country], file_path, None]
        else:
            anns = [None, [0], [country], file_path, None]


        return image, anns

#Works on CelebA face dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
class CelebADataset(data.Dataset):
    
    def __init__(self, transform):
        self.transform = transform
        self.img_folder = 'celeba'
        self.annotations_folder = 'Anno'
        
        self.image_ids = []
        #Adds the title of the image as its ID (e.g. 000006.jpg = 000006)
        with open('Anno/identity_CelebA.txt') as f:
            for line in f:
                stripped_line = line.strip()
                stripped_line = stripped_line.split()
                self.image_ids.append(stripped_line[0])
        
        print("done with ids (1/4 dataset steps)")
        self.attribute_names = ["Female", "Male"]

        # List of all the labels (e.g. Young, mustache)
        count = 0
        with open('Anno/list_attr_celeba.txt') as f:
            for line in f:
                stripped_line = line.strip()
                stripped_line = stripped_line.split()
                self.categories =stripped_line
                count += 1
                if count == 2:
                    break
        
        #category name is just equal to that category name ("young" = "young")
        self.labels_to_names = {}
        for category in self.categories:
            self.labels_to_names[category] = category
        print("done with categories (2/4 dataset steps)")
        
        self.scene_mapping = NoneDict()
        if os.path.exists('dataloader_files/celeba_scene_mapping.pkl'):
            self.scene_mapping = pickle.load(open('dataloader_files/celeba_scene_mapping.pkl', 'rb'))
        else:
            setup_scenemapping(self, 'celeba')
        
        print("done with scene mapping (3/4 dataset steps)")
        self.group_mapping = None

        #Note: This refers to labels that describe people in images
        self.people_labels = []
        
        #Needs to exist for gender analysis
        self.num_attribute_images = [0, 0]
        count = 0
        with open('Anno/list_attr_celeba.txt') as f:
            for line in f:
                if count >= 2:
                    stripped_line = line.strip()
                    stripped_line = stripped_line.split()
                    image_values = stripped_line[1:]
                    gender = int(image_values[20])
                    if gender > 0:
                        self.num_attribute_images[1] += 1
                    else:
                        self.num_attribute_images[0] += 1
                count += 1
                
        print("done with gender counting (4/4 dataset steps)")
        print(self.num_attribute_images)
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_path = self.img_folder + '/' + image_id
        return self.from_path(file_path)

    def __len__(self):
        return len(self.image_ids)

    def from_path(self, file_path):
        image_id = os.path.basename(file_path)

        image = Image.open(file_path).convert("RGB")
        image = self.transform(image)
        image_size = list(image.size())[1:]

        country = None

        image_anns = []

        #For each image, get gender and category information
        with open('Anno/list_attr_celeba.txt') as f:
            for line in f:
                stripped_line = line.strip()
                if image_id in stripped_line:
                    stripped_line = stripped_line.split()
                    image_values = stripped_line[1:]
                    #From values, gender value is at index 20
                    gender = int(image_values[20])
                    #For all categories, add an annotation for that category if value is 1 for particular image
                    for category in range(len(self.categories)):
                        if int(image_values[category]) == 1:
                            image_anns.append({'label':self.categories[category]})
                    break
        #For each image, get bbox information and add to bbox_digits
        with open('Anno/list_bbox_celeba.txt') as bbox_f:
            for line in bbox_f:
                stripped_line = line.strip()
                if image_id in stripped_line:
                    stripped_line = stripped_line.split()
                    bbox = stripped_line[1:]
                    
                    #x,y,width,height
                    #Normalize digits in same way as done in Coco
                    bbox_digits = [int(bbox[0]) / image_size[1], (int(bbox[0])+int(bbox[2])) / image_size[1], int(bbox[1]) / image_size[0], (int(bbox[1])+int(bbox[3])) / image_size[0]]
                    break
        
        #Females are marked as 1, and males as 0
        if gender < 0:
            gender = 1
        else:
            gender = 0
        
        gender_info = [[gender], [bbox_digits]]
       
        scene_group = self.scene_mapping[file_path]
        #Note: Gender info should not be in array since gender_info is already array
        anns = [image_anns, gender_info, [country], file_path, scene_group]

        return image, anns

'''
Dataset can be downloaded here: 
https://www.cityscapes-dataset.com/downloads/

After creating an account, you download the image data: 
gtFine_trainvaltest.zip (241MB) [md5]

and the gps data:
vehicle_trainvaltest.zip (2MB) [md5]
'''
class CityScapesDataset(data.Dataset):

    def __init__(self, transform): 
        self.transform = transform
        self.img_folder = '/Users/home/Desktop/research/data/cityscapes/gtFine_trainvaltest/gtFine/train'

        # directory storing gps information
        self.gps_folder = '/Users/home/Desktop/research/data/cityscapes/vehicle_trainvaltest/vehicle/train'

        # boundary shapefile
        with open("/Users/home/Downloads/stanford-nh891yz3147-geojson.json") as f:
            self.geo_boundaries = json.load(f)

        # csv data for choropleth analysis
        self.choropleth_filepath = "/Users/home/Downloads/data.csv"
        

        # store all of the city names in array [aachen, bochum, etc]
        self.city_names = os.listdir(self.gps_folder)
        self.city_names.remove('.DS_Store')

        # Adds the title of the image as its ID 
        # (e.g. aachen/aachen_000000_000019_gtFine_color.png = aachen/aachen_000000_000019)
        self.image_ids = []
        for city in self.city_names:
            filename_path = os.path.join(self.gps_folder, city)
            city_filenames = os.listdir(filename_path)
            self.image_ids = self.image_ids +  [os.path.join(city, name.split("_vehicle")[0]) for name in city_filenames]
        print("done with ids (1/2)")

        self.categories = ['unlabeled',
            'ego vehicle',
            'rectification border',
            'out of roi',
            'static',
            'dynamic',
            'ground',
            'road',
            'sidewalk',
            'parking',
            'rail track',
            'building',
            'wall',
            'fence',
            'guard rail',
            'bridge',
            'tunnel',
            'pole',
            'polegroup',
            'traffic light',
            'traffic sign',
            'vegetation',
            'rider',
            'truckgroup',
            'terrain',
            'sky',
            'person',
            'persongroup',
            'bicyclegroup',
            'motorcyclegroup',
            'rider',
            'ridergroup',
            'car',
            'cargroup',
            'truck',
            'bus',
            'caravan',
            'trailer',
            'train',
            'motorcycle',
            'bicycle',
            'license plate']
        self.labels_to_names = {i : i for i in self.categories}
        print("done with categories (2/2)")

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        return self.from_path(image_id)
    
    def __len__(self):
        return len(self.image_ids)
    
    def from_path(self, file_path):
        image_id = os.path.join(self.img_folder, "{0}_gtFine_color.png".format(file_path))
        image = Image.open(image_id).convert("RGB")
        image = self.transform(image)
        country = None
        # for each image, get category information
        image_anns = []

        category_path = os.path.join(self.img_folder, "{0}_gtFine_polygons.json".format(file_path))
        category_data = json.load(open(category_path)).get('objects', [])

        for i in range(len(category_data)):
            label = category_data[i].get('label', None)
            if label is not None:
                image_anns.append({'label': label})

        # for each image, get long lat gps information
        lat_lng = {}
        json_data = json.load(open(os.path.join(self.gps_folder, "{0}_vehicle.json".format(file_path))))
        if json_data is not None and "gpsLatitude" in json_data and "gpsLongitude" in json_data:
            lat_lng['lat'] = json_data["gpsLatitude"]
            lat_lng['lng'] = json_data["gpsLongitude"]

        anns = [image_anns, None, [country, lat_lng], file_path, None]    
        return image, anns