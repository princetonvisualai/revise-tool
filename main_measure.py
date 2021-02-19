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
    parser.add_argument('--measurements', nargs='+', type=str, default='obj_cnt',
            help='in map below')
    parser.add_argument('--dataset', type=str, default='openimages',
            help='input dataset to format')
    parser.add_argument('--folder', type=str, default='random',
            help='folder to store results in')
    parser.add_argument('--ngpu', type=int, default=0,
            help='number of gpu')

    args = parser.parse_args()

    index_to_measurement = {
        'obj_cnt': object_based.count_cooccurrence,
        'att_siz': gender_based.size_and_distance,
        'att_cnt': gender_based.count_cooccurrence,
        'att_dis': gender_based.distance_for_instance,
        'att_clu': gender_based.cluster_for_instance,
        'geo_ctr': geography_based.count_country, 
        'geo_tag': geography_based.count_tags,
        'obj_siz': object_based.supercategory_size_and_distance,
        'obj_ppl': object_based.supercategory_with_people,
        'obj_scn': object_based.scene_categorization,
        'geo_lng': geography_based.count_langs,
        'att_scn': gender_based.scenes
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
    elif args.dataset == 'celeba':
        dataset = CelebADataset(transform_train)
    
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
