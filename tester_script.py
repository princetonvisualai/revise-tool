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
import pickle


NUM_EXS=5

def validate_dataset(dataset): 

    # set up dataset
    try: 
        print('starting setup')
        t = transforms.Compose([
         transforms.CenterCrop(10),
        transforms.ToTensor(),
        ])
        ds = dataset(t) 
        print('end setup')
    except Exception as e: 
        print('ERROR: Initialization failed before testing:', e)
        sys.exit()
    print('testing imageids')
    # testing image_ids
    try: 
        ds.image_ids 
    except AttributeError: 
        print('ERROR: self.image_ids is a required field.')  
    if not isinstance(ds.image_ids, list): 
        print('---- Image_ids must be of type: list ----')
        print('ERROR: Currently of type:', type(ds.image_ids))
    else:
        print('\n--- Number of images ---')
        print(str(len(ds.image_ids)), '\n')
        rand_inds = random.sample(  range(len(ds.image_ids) - 1), NUM_EXS  )


    # testing categories
    try: 
        ds.categories
    except AttributeError: 
        print('ERROR: self.categories is a required field.')
    if not isinstance(ds.categories, list):
        print('--- Categories must be type: list ---')
        print('ERROR: Currently of type:', type(ds.categories), '\n')
    else: 
        print('---- Total number of labels in the dataset ---')
        print( str(len(ds.categories)),  '\n')


    # testing scene mappings 
    try:
        if not isinstance(ds.scene_mapping, dict):
            print('--- Scene_mapping must be type: dict ---')
            print('ERROR: Currently of type:', type(ds.scene_mapping), '\n')
    except AttributeError:
       pass 


    # testing supercategories_to_names
    try:
        if not isinstance(ds.supercategories_to_names, dict): 
            print('ERROR: self.supercategories_to_names must be type: dict \n')
    except AttributeError:
        print('ERROR: self.supercategories_to_names is a required field.')
        print('Please set self.categories_to_names = DEFAULT_GROUPINGS_TO_NAMES \n')

    # testing if group_mappings exists
    try: 
        ds.group_mapping
    except AttributeError: 
        print('ERROR: self.group_mapping is a required field.')


    # testing labels_to_names
    try: 
        ds.labels_to_names
    except AttributeError: 
        print('ERROR: self.labels_to_names is a required field.')
    if not isinstance(ds.labels_to_names, dict): 
        print('---- Labels_to_names must be type: dict ---') 
        print('ERROR: Currently of type:', type(ds.labels_to_names), '\n')
    else: 

        print('---', str(NUM_EXS), 'random examples of [label] -> [supercategory] ---' )
        len_labels = len(ds.labels_to_names) if len(ds.labels_to_names)!=0 else len(ds.categories)
        rand_inds = random.sample(  range(len_labels - 1), NUM_EXS  )
        for rand_ind in rand_inds:  
  
            try:
                supercat = ds.group_mapping(  list(ds.labels_to_names.items())[rand_ind][0]  )
                supercat_name = ds.supercategories_to_names[supercat]
                print(  ds.labels_to_names[ds.categories[rand_ind]], '->', supercat_name)
                scflag=True
            except Exception as e:
                print( ds.labels_to_names[ds.categories[rand_ind]] )
                scflag=False
        if not scflag: print('ERROR: self.supercategories not set up correctly so supercategories not displayed.')
        print('\n')


    
    # testing __len__:
    try: 
        ds.__len__()
        if ds.__len__() != len(ds.image_ids):
            print('ERROR: self.__len__() must be equal to length of self.image_ids')
            print('self.__len__() returns', ds.__len__(), '/ length of self.image_ids =', len(ds.image_ids), '\n')
    except AttributeError: 
        print('ERROR: self.__len__() is a required method.\n')
 

    # testing __getitem__ and from_path
    rand_ind = random.randint(0, ds.__len__()-1)
    try: 
        x = ds.__getitem__(rand_ind)
    except AttributeError: 
        print('ERROR: self.__getitem___() is a required method.\n')
   
    x = ds.__getitem__(rand_ind)
    if len(x) != 2: 
        print('ERROR: self.__getitem__() must return a tuple of length: 2')
        print('Return value should be in form (image, annotations)\n')
        sys.exit()
    img, anns = x
    if not isinstance(anns, list): 
        print('ERROR: Annotations must be of type: list\n') 
        sys.exit()
    if len(anns) != 5:
        print('ERROR: self.__getitem__() should return annotations of length: 5')
        print('Annotations must be a list containing [image_anns, gender_info, [country, lat_lng], file_path, scene_group]\n')
        sys.exit()

    labels, att, geo, fp, scn = anns

    if len(labels) > 1: 
        if not isinstance(labels[0], dict) or labels[0].get('label', None)==None: 
            print('ERROR: image_anns must be a list of dicts. If there are >0 dicts, must contain keyword \'label\' \n')
    for label in labels: 
        wrong_bbox=None
        if label.get('bbox', None): 
            for coord in label['bbox']: wrong_bbox=label['bbox'] if (coord<0 or coord>1) else None
    if wrong_bbox: 
        print('ERROR: All bounding box numbers must be scaled between 0 and 1. Got bounding box: ', end='')
        for coord in wrong_bbox: 
            print('%.2f, ' % coord, end='')
        print('\n')

    if att and not att[0]:
        print('ERROR: If no attribute annotations, must be an empty list i.e. [], got:', att, '\n')
    if att and not isinstance(att[0], list): 
        print('ERROR: Attribute annotation must be in a list, got:', att[0], '\n')
    elif att and isinstance(att[0], list):
        if len(att)==2 and len(att[0])!=len(att[1]): print('ERROR: length of annotation list is not equal to length of bbox list.\n')
        try: 
            for a in att[0]:
                if a >= len(ds.attribute_names): print('ERROR: attribute annotation out of index for given self.attribute_names. Got value:', a, '\n')
        except Exception as e:
            print(e)
            print('ERROR: self.attribute_names is a required for attribute annotations.')

    if len(geo) !=1 and len(geo) != 2: 
        print('ERROR: geography info must be a list of length: 1 in the form [None] or [country] or length: 2 in the form [country, lat_lng] \n')
    if len(geo) ==2 and (not isinstance(geo[1], dict) or len(geo[1])!= 2 or not geo[1].get('lat') or not geo[1].get('lng')): 
        print('ERROR: lat_lng in [country, lat_lng] must be of type:dict with 2 keys: \'lat\' and \'lng\' \n')
     

    rand_inds = random.sample(  range(ds.__len__() - 1), NUM_EXS  )
    print('--- View folder "tester_script_out" for images ----')
    for i in range(NUM_EXS):
        img, anns = ds.__getitem__(rand_inds[i]) 
        img = img.permute(1, 2, 0).numpy()
        cv2.imwrite('tester_script_out/example_' + str(i) + '.jpg', img)                 #need to generalize 
        print('--- Annotations for example_' + str(i) + '.jpg in tester_script_out ---') 

        labels, attribute, geography, filepath, scene = anns

        if not labels or len(labels) == 0: 
            print('Label: No annotations for this image', end='')
        for label in labels: 
            curr_label = ds.labels_to_names.get(label['label'], label['label']) if len(ds.labels_to_names)!=0 else label['label']
            print('Label:', str(curr_label), end='')

            if label.get('bbox'): 
                print(', bbox: ', end='')
                for coord in label['bbox']:
                    print("%.2f, " % coord, end='')
            print('')


        if not attribute or not attribute[0]: print('Attribute: No annotation for this image', end='')
        else: 
            print('Attribute: ', end='')
            atts = attribute[0]
            bboxs = attribute[1] if len(att)>1 else None
            for i in range(len(atts)):
                print(str(ds.attribute_names[atts[i]]), end='')
                if bboxs: 
                    print(', bbox: ', end='')
                    for coord in bboxs[i]: 
                        print('%.2f, ' % coord, end='')
        print('')


        for geo in geography: 
            print('Geography:', geo) if geo else print('Geography: No annotations for this image')

        print('Filepath:', filepath)

        if not scene or not scene[0]: 
            print('Scene group: No annotations for this image')
        else:
            try:
               info = pickle.load(open('util_files/places_scene_info.pkl', 'rb'))
               idx_to_scene = info['idx_to_scene']
               idx_to_scenegroup = info['idx_to_scenegroup']
               for scn in scene: 
                   print('Scene group: ', idx_to_scenegroup[scn])
            except Exception as e:
               print('ERROR: Must have file util_files/places_scene_info.pkl for scene mapping.')
               print('Exception: ', e)
               for scn in scene: 
                   print('Scene group: ', scn)
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

 
