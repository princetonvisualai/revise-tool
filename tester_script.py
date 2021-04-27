# #!/usr/bin/env python3
import sys 
import importlib
import torchvision.transforms as transforms
import torch
import os
import random
import numpy as np
import cv2
from operator import itemgetter


NUM_EXS=3

def validate_dataset(dataset): 

    # set up dataset
    try: 
        t = transforms.Compose([
         transforms.CenterCrop(10),
        transforms.ToTensor(),
        ])
        ds = dataset(t)
    except Exception as e: 
        print('ERROR: Initialization failed before testing:', e)
        sys.exit()

    # testing img_folder 
    try: 
        if os.path.isdir(ds.img_folder):
            print('\n ---- Image folder identified as ----') 
            print(ds.img_folder, '\n')
            cmd = 'ls -1 ' + ds.img_folder + ' | wc -l'
            print('---- Total number of objects found in image folder ----')
            print(str(os.popen(cmd).read()).strip(), '\n')
        else: 
            print('---- Image folder not detected ---- \n')
    except AttributeError: 
        pass

    # testing image_ids and printing images 
    if not isinstance(ds.image_ids, list): 
        print('---- Image_ids must be type: list ----')
        print('ERROR: Currently of type:', type(ds.image_ids))
    else:
        print('--- Total number of image_ids ---')
        print(str(len(ds.image_ids)), '\n')
        rand_inds = random.sample(  range(len(ds.image_ids) - 1), NUM_EXS  )

    # testing labels 
    if not isinstance(ds.labels_to_names, dict): 
        print('---- Labels_to_names must be type: dict ---') 
        print('ERROR: Currently of type:', type(ds.labels_to_names), '\n')
    else: 
        print('--- Total number of label mappings ---')
        print( str(len(ds.labels_to_names)), '\n')
        if len(ds.labels_to_names) != 0:
            print('---', str(NUM_EXS), 'random examples of human-interpretable labels in dataset ---' )
            rand_inds = random.sample(  range(len(ds.labels_to_names) - 1), NUM_EXS  )
            for rand_ind in rand_inds: 
                print(  list(ds.labels_to_names.items())[rand_ind][1]   )
            print('\n')

    # testing categories
    if not isinstance(ds.categories, list):
        print('--- Categories must be type: list ---')
        print('ERROR: Currently of type:', type(ds.categories), '\n')
    else: 
        print('---- Total number of categories in dataset ----')
        print(str(len(ds.categories)), '\n')

    # testing scene mappings 
    try:
        if not isinstance(ds.scene_mapping, dict):
            print('--- Scene_mapping must be type: dict ---')
            print('ERROR: Currently of type:', type(ds.scene_mapping), '\n')
    except AttributeError:
       print('--- Scene_mapping currently not set up for use ---')
       print('Please refer to dataloader documentation in "datasets.py" for advice on setup. \n')
    
    # testing __len__
    print('--- __len__ method returns ----')
    print(ds.__len__())
    if ds.__len__() != len(ds.image_ids): 
        print('ERROR: __len__() should return:', len(ds.image_ids))
    print('\n')

    # testing __getitem__
    images_list = os.listdir(ds.img_folder)
    rand_inds = random.sample(  range(len(images_list) - 1), NUM_EXS  )
    print('--- Pulling images using these random indices ---')
    print(rand_inds, '\n')
    print('--- View folder "tester_script_out" for images ----')
    for i in range(NUM_EXS):
        img, anns = ds.__getitem__(rand_inds[i]) 
        img = img.permute(1, 2, 0).numpy()
        cv2.imwrite('tester_script_out/example_' + str(i) + '.jpg', img) #need to generalize 
        print('-- sample annotations for example_' + str(i) + '.jpg in tester_script_out --')
        try: 
            labels, attribute, geography, filepath, scene = anns
            for label in labels: 
                curr_label = ds.labels_to_names.get(label['label'], label['label']) if len(ds.labels_to_names)!=0 else label['label']
                print('Label:', curr_label + ', bbox: ' + str(label['bbox'])) if label.get('bbox', None) else print('Label:', label['label'])
            for att in attribute: 
                print('Attribute:', att)
            for geo in geography: 
                print('Geography:', geo)
            print('Filepath:', filepath)
            print('Scene group:', scene)
        except AttributeError, IndexError, TypeError: 
            print('ERROR: image_anns should be of type: dict, with key \'label\' ') 
        except ValueError: 
            print('ERROR: Annotations list should be of length 5.')
            print('Annotations should be in form [image_anns, gender_info, [country, lat_lng], file_path, scene_group]')
            print(anns)
        print('\n')

if __name__ == "__main__": 
    if len(sys.argv) != 2: 
        print("usage: tester_script.py [DatasetName]")
        sys.exit()

    filename = 'datasets'          
    module = sys.argv[1]

    try: 
        dataset = getattr(importlib.import_module(filename), module)
    except AttributeError as e: 
        print('error: no class', module, 'in file datasets.py')
        sys.exit()

    validate_dataset(dataset)

 
