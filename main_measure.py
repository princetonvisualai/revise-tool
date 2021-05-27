import argparse
from datasets import *
import pickle
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from measurements import object_based
from measurements import attribute_based
from measurements import geography_based
from measurements import attribute_based

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
        'obj_cnt': object_based.obj_cnt,
        'att_siz': attribute_based.att_siz,
        'att_cnt': attribute_based.att_cnt,
        'att_dis': attribute_based.att_dis,
        'att_clu': attribute_based.att_clu,
        'geo_ctr': geography_based.geo_ctr, 
        'geo_tag': geography_based.geo_tag,
        'obj_siz': object_based.obj_siz,
        'obj_ppl': object_based.obj_ppl,
        'obj_scn': object_based.obj_scn,
        'geo_lng': geography_based.geo_lng,
        'att_scn': attribute_based.att_scn,
        'geo_att': geography_based.geo_att
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
    elif args.dataset == 'cityscapes':
        dataset = CityScapesDataset(transform_train)
    
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
