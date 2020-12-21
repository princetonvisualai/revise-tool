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
    model_file = '%s_places365.pth.tar' % arch
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

GROUPINGS_TO_NAMES = {
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

class TemplateDataset(data.Dataset):
    
    def __init__(self, transform):
        self.transform = transform
        
        # Where the images are located (doesn't need to exist, but can be helpful for other functions)
        self.img_folder = ''

        # List of all of the image ids
        self.image_ids = [] 

        # Maps label to the human-readable name
        self.labels_to_names = {}

        # List of all the labels
        self.categories = []

        # Maps from filepath to scenes
        # Can be set up by running AlexNet Places365 model by running the following command:
        # self.scene_mapping = setup_scenemapping(self, '[name of dataset]')
        self.scene_mapping = NoneDict()

        # Maps each label to number of supercategory group, which is listed in keys of GROUPINGS_TO_NAMES (optional)
        self.group_mapping = None

        # Labels that correspond to people (optional)
        self.people_labels = []

        # Number of images from dataset that are female and male (optional, doesn't need to exist)
        self.num_gender_images = [0, 0]
        
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

        person_bbox = None # optional
        gender = None # optional, we have used 0 for female and 1 for male when these labels exist
        gender_info = [gender, person_bbox] # optional

        country = None # optional

        image_anns = None
        scene_group = self.scene_mapping[file_path] # optional
        anns = [image_anns, [gender_info], [country], file_path, scene_group]

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
            self.num_gender_images = info['num_gender']
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

            self.num_gender_images = [0, 0]
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
                    self.anns[key] = [self.anns[key], [2, biggest_bbox], [0]]
                    self.num_gender_images[1] += 1
                elif w_presence > 0 and m_presence == 0:
                    self.anns[key] = [self.anns[key], [1, biggest_bbox], [0]]
                    self.num_gender_images[0] += 1
                else:
                    self.anns[key] = [self.anns[key], [0], [0]]
            info = {}
            info['anns'] = self.anns
            info['num_gender'] = self.num_gender_images
            pickle.dump(info, open('dataloader_files/openimage_anns.pkl', 'wb'))

class CoCoDataset(data.Dataset):

    def __init__(self, transform):
        self.transform = transform
        
        self.img_folder = 'Data/Coco/2014data/train2014'
        self.coco = COCO('Data/Coco/2014data/annotations/instances_train2014.json')
        gender_data = pickle.load(open('Data/Coco/2014data/bias_splits/train.data', 'rb'))
        self.gender_info = {int(chunk['img'][15:27]): chunk['annotation'][0] for chunk in gender_data}

        ids = list(self.coco.anns.keys())
        self.image_ids = list(set([self.coco.anns[this_id]['image_id'] for this_id in ids]))

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.labels_to_names = {}
        for cat in cats:
            self.labels_to_names[cat['id']] = cat['name']

        self.categories = list(self.labels_to_names.keys())

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
        self.num_gender_images = [6642, 16324]
        
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
        if biggest_bbox != 0 and image_id in self.gender_info.keys():
            anns = [formatted_anns, [self.gender_info[image_id] + 1, biggest_bbox], [0], file_path, scene]
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
        if biggest_bbox != 0 and image_id in self.gender_info.keys():
            anns = [formatted_anns, [self.gender_info[image_id] + 1, biggest_bbox], [0], file_path, scene]
        else:
            anns = [formatted_anns, [0], [0], file_path, scene]

        return image, anns

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


        self.group_mapping = None
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
    
    def __init__(self, transform, metric=0):
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

        if metric in [0, 6]:
            self.version = 'intersect' # has tags from cleaned English, and geolocation
        elif metric in [10]:
            self.version = 'alllang' # has tags in any language, and geolocation
        elif metric in [5]:
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
