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
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


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
        print('initialization failed before testing: error: ', e)
        sys.exit()

    # testing img_folder 
    if os.path.isdir(ds.img_folder): 
        print('img_folder: '+ds.img_folder+' path found.')
        cmd = 'ls -1 ' + ds.img_folder + ' | wc -l'
        print('number of objects found in img_folder: ' + str(os.popen(cmd).read()).strip())
    else: 
        print('img_folder: '+ds.img_folder+' does not exist or not a directory.')
        sys.exit()

    # testing image_ids and printing images 
    if not isinstance(ds.image_ids, list): 
        print('self.image_ids must be type: list')
    else: 
        print('number of image ids: ' + str(len(ds.image_ids)))
        rand_inds = random.sample(  range(len(ds.image_ids) - 1), NUM_EXS  )
        print('some random image_ids: ')
        for i in range(NUM_EXS):
            print(ds.image_ids[rand_inds[i]])

    # testing labels 
    if not isinstance(ds.labels_to_names, dict): 
        print('self.labels_to_names must be type: dict') 
        sys.exit()
    else: 
        print('number of label mappings: ' + str(len(ds.labels_to_names)))
        if len(ds.labels_to_names) != 0:
            print('some random examples of label to name mappings: ')
            rand_inds = random.sample(  range(len(ds.labels_to_names) - 1), NUM_EXS  )
            for rand_ind in rand_inds: 
                print(  list(ds.labels_to_names.items())[rand_ind]   )
    
    # testing categories
    if not isinstance(ds.categories, list):
        print('self.categories must be type: list')
        sys.exit()
    else: 
        print('number of total categories: ' + str(len(ds.categories)))
        if len(ds.categories) != 0:
            print('some random examples of categories: ')
            rand_inds = random.sample(  range(len(ds.categories) - 1), NUM_EXS  )
            for rand_ind in rand_inds: 
                print( ds.categories[rand_ind]  )

    # testing scene mappings 
    try:
        if not isinstance(ds.scene_mapping, dict): 
            print('self.scene_mapping must be type: dict')
    except AttributeError:
       print('nothing set up currently for ds.scene_mapping.')
    
    # testing __len__
    print('self.__len__() =', ds.__len__())

    # testing __getitem__
    images_list = os.listdir(ds.img_folder)
    rand_inds = random.sample(  range(len(images_list) - 1), NUM_EXS  )
    print('checking random indices:', rand_inds)
    print('check folder tester_script_out for the indexed images.')
    for i in range(NUM_EXS):
        img, anns = ds.__getitem__(rand_inds[i]) 
        img = img.permute(1, 2, 0).numpy()
        cv2.imwrite('tester_script_out/index_' + str(i) + '.jpg', img) #need to generalize 
        print('indexed image ' + str(rand_inds[i]) + ' annotations:')
        print(anns)



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

 
