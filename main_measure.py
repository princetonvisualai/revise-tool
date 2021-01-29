import argparse
from datasets import *
import pickle
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import object_based
import gender_based
import geography_based

def main():

    if not os.path.exists("dataloader_files"):
        os.mkdir("dataloader_files")
    if not os.path.exists("results"):
        os.mkdir("results")

    parser = argparse.ArgumentParser(description='Measurement')
    parser.add_argument('--measurements', nargs='+', type=int, default=0,
            help='in map below')
    parser.add_argument('--dataset', type=str, default='openimages',
            help='input dataset to format')
    parser.add_argument('--folder', type=str, default='random',
            help='folder to store results in')
    parser.add_argument('--ngpu', type=int, default=0,
            help='number of gpu')

    args = parser.parse_args()


    index_to_measurement = {
        0: object_based.count_cooccurrence,
        1: gender_based.size_and_distance,
        2: gender_based.count_cooccurrence,
        3: gender_based.distance_for_instance,
        4: gender_based.cluster_for_instance,
        5: geography_based.count_country, 
        6: geography_based.count_tags,
        7: object_based.supercategory_size_and_distance,
        8: object_based.supercategory_with_people,
        9: object_based.scene_categorization,
        10: geography_based.count_langs,
        11: gender_based.scenes
    }


    transform_train = transforms.Compose([           
        transforms.ToTensor(),                          
        ])

    if not os.path.exists("results/" + args.folder):
        os.mkdir("results/" + args.folder)

    if args.dataset == 'openimages':
        dataset = OpenImagesDataset(transform_train)
    elif args.dataset == 'coco':
        dataset = CoCoDataset(transform_train)
    elif args.dataset == 'sun':
        dataset = SUNDataset(transform_train)
    elif args.dataset == 'yfcc':
        dataset = None
    elif args.dataset == 'imagenet':
        dataset = ImagenetDataset(transform_train)
    
    for meas in args.measurements:
        print("Starting measurement {}".format(meas))
        if args.dataset == 'yfcc':
            dataset = YfccPlacesDataset(transform_train, meas)
            dataloader = data.DataLoader(dataset=dataset, 
              num_workers=0,
              batch_size=1,
              collate_fn=collate_fn,
              shuffle=True)
        else:
            dataloader = data.DataLoader(dataset=dataset, 
              num_workers=0,
              batch_size=1,
              collate_fn=collate_fn,
              shuffle=True)
        index_to_measurement[meas](dataloader, args)
        print("Finished measurement {}".format(meas))

if __name__ == '__main__':
    main()
