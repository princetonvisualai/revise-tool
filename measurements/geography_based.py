import os
assert 'measurement' not in os.getcwd() and 'analysis_notebooks' not in os.getcwd(), "Script must be run from home directory"
import numpy as np
import pickle
import fasttext
from collections import Counter
import re
from countryinfo import CountryInfo
import torchvision.transforms as transforms
from torchvision import models
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import torch
import pycountry
import copy
from tqdm import tqdm
import json
from shapely.geometry import Point
from shapely.geometry import shape 
from shapely.geometry import Polygon
import os

def country_to_iso3(country):
    missing = {'South+Korea': 'KOR',
            'North+Korea': 'PRK',
            'Laos': 'LAO',
            'Caribbean+Netherlands': 'BES',
            'St.+Lucia': 'LCA',
            'East+Timor': 'TLS',
            'Democratic+Republic+of+Congo': 'COD',
            'Swaziland': 'SWZ',
            'Cape+Verde': 'CPV',
            'C%C3%B4te+d%C2%B4Ivoire': 'CIV',
            'Ivory+Coast': 'CIV',
            'Channel+Islands': 'GBR'
            }
    try:
        iso3 = pycountry.countries.search_fuzzy(country.replace('+', ' '))[0].alpha_3
    except LookupError:
        try:
            iso3 = missing[country]
        except KeyError:
            iso3 = None
    return iso3

def geo_ctr(dataloader, args):
    # redirect to geo_ctr_gps if dataset is of gps form:
    if (dataloader.dataset.geography_info_type == "GPS_LABEL"):
        print("redirecting to geo_ctr_gps()...")
        return geo_ctr_gps(dataloader, args)
    if (dataloader.dataset.geography_info_type == "STRING_FORMATTED_LABEL" and dataloader.dataset.geography_label_string_type == "REGION_LABEL"):
       print("redirecting to geo_ctr_region()...")
       return geo_ctr_region(dataloader, args)
    
    print("starting geo_ctr() for country label format")
    counts = {}

    for i, (data, target) in enumerate(tqdm(dataloader)):
        country = target[2][0]
        if country not in counts.keys():
            counts[country] = 0
        counts[country] += 1

    pickle.dump(counts, open("results/{}/geo_ctr.pkl".format(args.folder), "wb"))

# private function called from geo_counter() if dataset is of region label form
def geo_ctr_region(dataloader, args):
    region_to_id_map = {}
    id_to_region_map = {}

    for i, (data, target) in enumerate(tqdm(dataloader)):
        region = target[2][0]
        id_to_region_map[target[3]] = region
        if region not in region_to_id_map.keys():
            region_to_id_map[region] = 0
        region_to_id_map[region] += 1

    combined_dict = {}
    combined_dict["region_to_id"] = region_to_id_map
    combined_dict["id_to_region"] = id_to_region_map

    pickle.dump(combined_dict, open("results/{}/geo_ctr.pkl".format(args.folder), "wb"))

# private function called from geo_ctr() if dataset is of gps form
def geo_ctr_gps(dataloader, args):
    # import custom political boundaries shapefile from dataset 
    geo_boundaries = dataloader.dataset.geo_boundaries
    geo_boundaries_key_name = dataloader.dataset.geo_boundaries_key_name

    # import subregion boundaries shapefile
    subregion_boundaries = dataloader.dataset.subregion_boundaries
    subregion_boundaries_key_name = dataloader.dataset.subregion_boundaries_key_name

    # fn that returns name of political region that a point falls into (eg. Manhattan)
    def bin_point(lng, lat, is_subregion):
        point = Point(lng, lat)
        # check each polygon to see if it contains the point
        if not is_subregion:
            for feature in geo_boundaries['features']:
                polygon = shape(feature['geometry'])
                if polygon.contains(point):
                    return feature['properties'][geo_boundaries_key_name]
            return None
        else:
            # subregion binning
            for feature in subregion_boundaries['features']:
                polygon = shape(feature['geometry'])
                if polygon.contains(point):
                    return feature['properties'][subregion_boundaries_key_name]
            return None
    
    # maps each region (eg. Manhattan) to an array of id's representing the data filename
    region_to_id_map = {}
    # maps each id (representing data filename) to gps (lat + lng information)
    id_to_gps_map = {}
    # maps each id (representing data filename) to region 
    id_to_region_map = {}

    # maps each subregion (eg. North America) to an array of id's representing the data filename
    subregion_to_id_map = {}
    id_to_subregion_map = {}
    
    for i, (data, target) in enumerate(tqdm(dataloader)):
        lat_lng = target[2][1]
        id_to_gps_map[target[3]] = lat_lng
        # find which region the image was taken from
        region_name = bin_point(lat_lng['lng'], lat_lng['lat'], False)

        # add filepath id to region_to_id_map
        if region_name is not None:
            id_to_region_map[target[3]] = region_name
            id_list = region_to_id_map.get(region_name, [])
            id_list.append(target[3])
            # add filepath id to region_to_id_map
            region_to_id_map[region_name] = id_list

        else:
            id_to_region_map[target[3]] = 'out_of_boundary'
            id_list = region_to_id_map.get('out_of_boundary', [])
            id_list.append(target[3])
            # add filepath id to region_to_id_map
            region_to_id_map['out_of_boundary'] = id_list

        if subregion_boundaries is not None:
            subregion_name = bin_point(lat_lng['lng'], lat_lng['lat'], True)
            if subregion_name is not None:
                id_to_subregion_map[target[3]] = subregion_name
                id_list = subregion_to_id_map.get(subregion_name, [])
                id_list.append(target[3])
                subregion_to_id_map[subregion_name] = id_list
            else:
                id_list = subregion_to_id_map.get('out_of_boundary', [])
                id_list.append(target[3])
                subregion_to_id_map['out_of_boundary'] = id_list
                id_to_subregion_map[target[3]] = "out_of_boundary"

    # combine all the maps into one big one
    counts_gps = {}
    counts_gps['region_to_id'] = region_to_id_map
    counts_gps['id_to_gps'] = id_to_gps_map
    counts_gps['id_to_region'] = id_to_region_map
    # only create mapping if there are at least 2 unique subregions
    if len(subregion_to_id_map.keys()) >= 2:
        counts_gps['subregion_to_id'] = subregion_to_id_map
        counts_gps['id_to_subregion'] = id_to_subregion_map
    else:
        print("Not enough subregions (< 3) for subregion analysis")
    pickle.dump(counts_gps, open("results/{}/geo_ctr.pkl".format(args.folder), "wb"))

def geo_tag(dataloader, args):
    # redirect to geo_tag_gps if dataset is of gps form:
    if (dataloader.dataset.geography_info_type == "GPS_LABEL"):
        print("redirecting to geo_tag_gps()...")
        return geo_tag_gps(dataloader, args)
    elif (dataloader.dataset.geography_info_type == "STRING_FORMATTED_LABEL" and dataloader.dataset.geography_label_string_type == "REGION_LABEL"):
        print("redirecting to geo_tag_region()...")
        return geo_tag_region(dataloader, args)
    country_tags = {}
    tag_to_subregion_features = {}
    categories = dataloader.dataset.categories
    iso3_to_subregion = pickle.load(open('util_files/iso3_to_subregion_mappings.pkl', 'rb'))
    unique_subregions = set(list(iso3_to_subregion.values()))

    # Extracts features from model pretrained on ImageNet
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = models.alexnet(pretrained=True).to(device)
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    subregion_features = {}
    for subregion in unique_subregions:
        subregion_features[subregion] = []
    for cat in range(len(categories)):
        tag_to_subregion_features[cat] = copy.deepcopy(subregion_features)
    for i, (data, target) in enumerate(tqdm(dataloader)):
        if data is None:
            continue
        country = target[2][0]
        anns = target[0]
        filepath = target[3]
        this_categories = list(set([categories.index(ann['label']) for ann in anns]))
        subregion = iso3_to_subregion[country_to_iso3(country)]
        if country not in country_tags.keys():
            country_tags[country] = np.zeros(len(categories))
        this_features = None
        for cat in this_categories:
            if len(tag_to_subregion_features[cat][subregion]) < 500:
                data = normalize(data).to(device)
                big_data = F.interpolate(data.unsqueeze(0), size=224, mode='bilinear').to(device)
                this_features = model.forward(big_data)
                break
        for cat in this_categories:
            country_tags[country][cat] += 1
            if this_features is not None and len(tag_to_subregion_features[cat][subregion]) < 500:
                tag_to_subregion_features[cat][subregion].append((this_features.data.cpu().numpy(), filepath))

    info_stats = {}
    info_stats['country_tags'] = country_tags
    info_stats['tag_to_subregion_features'] = tag_to_subregion_features
    pickle.dump(info_stats, open("results/{}/geo_tag.pkl".format(args.folder), "wb"))

# private function called from geo_tag() if dataset is of gps form
def geo_tag_gps(dataloader, args):
    # map from a region name to a list whose value at index i represents count of category
    # i 
    region_tags = {}
    subregion_tags = None # initialize later if subregion dat is available
    tag_to_region_features = {}
    categories = dataloader.dataset.categories

    if not os.path.exists("results/{}/geo_ctr.pkl".format(args.folder)):
        print('running geo_ctr_gps() first to get necessary info...')
        geo_ctr_gps(dataloader, args)
    
    counts_gps = pickle.load(open("results/{}/geo_ctr.pkl".format(args.folder), "rb"))
    id_to_region = counts_gps['id_to_region']
    id_to_subregion = counts_gps.get("id_to_subregion", None)

    # get name of regions
    unique_regions = list(set(id_to_region.values()))

    # get name of subregions, if applicable
    unique_subregions = None
    if id_to_subregion is not None:
        subregion_tags = {}
        unique_subregions = list(set(id_to_subregion.values()))

    # Extracts features from model pretrained on ImageNet
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = models.alexnet(pretrained=True).to(device)
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    region_features = {}
    for region in unique_regions:
        region_features[region] = []

    for cat in range(len(categories)):
        tag_to_region_features[cat] = copy.deepcopy(region_features)

    for i, (data, target) in enumerate(tqdm(dataloader)):
        if data is None:
            continue
        region_name = id_to_region.get(target[3], None)
        if region_name is None:
            continue
        anns = target[0]
        filepath = target[3]
        this_categories = list(set([categories.index(ann['label']) for ann in anns]))

        if region_name not in region_tags.keys():
            region_tags[region_name] = np.zeros(len(categories))
        
        subregion_name = None
        if id_to_subregion is not None:
            subregion_name = id_to_subregion[target[3]]
            if subregion_name not in subregion_tags.keys():
                subregion_tags[subregion_name] = np.zeros(len(categories))

        this_features = None
        for cat in this_categories:
            if len(tag_to_region_features[cat][region_name]) < 500:
                data = normalize(data).to(device)
                big_data = F.interpolate(data.unsqueeze(0), size=224, mode='bilinear').to(device)
                this_features = model.forward(big_data)
                break
        for cat in this_categories:
            if this_features is not None and len(tag_to_region_features[cat][region_name]) < 500:
                tag_to_region_features[cat][region_name].append((this_features.data.cpu().numpy(), filepath))
        for ann in anns:
            region_tags[region_name][categories.index(ann['label'])] += 1
            if id_to_subregion is not None:
                subregion_tags[subregion_name][categories.index(ann['label'])] += 1
    info_stats = {}
    info_stats['region_tags'] = region_tags
    if id_to_subregion is not None:
        print("Adding subregion tags...")
        info_stats['subregion_tags'] = subregion_tags
    info_stats['tag_to_region_features'] = tag_to_region_features
    pickle.dump(info_stats, open("results/{}/geo_tag.pkl".format(args.folder), "wb"))

# private function called from geo_tag() if dataset is of STRING_FORMATTED_LABEL + REGION_LABEL form
def geo_tag_region(dataloader, args):
    # map from a region name to a list whose value at index i represents count of category
    # i 
    region_tags = {}
    tag_to_region_features = {}
    categories = dataloader.dataset.categories

    if not os.path.exists("results/{}/geo_ctr.pkl".format(args.folder)):
        print('running geo_ctr_region() first to get necessary info...')
        geo_ctr_region(dataloader, args)
    
    counts = pickle.load(open("results/{}/geo_ctr.pkl".format(args.folder), "rb"))
    id_to_region = counts_gps['id_to_region']

    # get name of regions
    unique_regions = list(set(id_to_region.values()))

    # Extracts features from model pretrained on ImageNet
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = models.alexnet(pretrained=True).to(device)
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    region_features = {}
    for region in unique_regions:
        region_features[region] = []

    for cat in range(len(categories)):
        tag_to_region_features[cat] = copy.deepcopy(region_features)

    for i, (data, target) in enumerate(tqdm(dataloader)):
        if data is None:
            continue
        region_name = id_to_region[target[3]]
        anns = target[0]
        filepath = target[3]
        this_categories = list(set([categories.index(ann['label']) for ann in anns]))

        if region_name not in region_tags.keys():
            region_tags[region_name] = np.zeros(len(categories))
        this_features = None
        for cat in this_categories:
            if len(tag_to_region_features[cat][region_name]) < 500:
                data = normalize(data).to(device)
                big_data = F.interpolate(data.unsqueeze(0), size=224, mode='bilinear').to(device)
                this_features = model.forward(big_data)
                break
        for cat in this_categories:
            if this_features is not None and len(tag_to_region_features[cat][region_name]) < 500:
                tag_to_region_features[cat][region_name].append((this_features.data.cpu().numpy(), filepath))
        for ann in anns:
            region_tags[region_name][categories.index(ann['label'])] += 1
    info_stats = {}
    info_stats['region_tags'] = region_tags
    info_stats['tag_to_region_features'] = tag_to_region_features
    pickle.dump(info_stats, open("results/{}/geo_tag.pkl".format(args.folder), "wb"))
    
def geo_lng(dataloader, args):
    mappings = pickle.load(open('util_files/country_lang_mappings.pkl', 'rb'))
    iso3_to_lang = mappings['iso3_to_lang']
    # Country to iso3 mappings that are missing
    missing = {'South+Korea': 'KOR',
            'North+Korea': 'PRK',
            'Laos': 'LAO',
            'Caribbean+Netherlands': 'BES',
            'St.+Lucia': 'LCA',
            'East+Timor': 'TLS',
            'Democratic+Republic+of+Congo': 'COD',
            'Swaziland': 'SWZ',
            'Cape+Verde': 'CPV',
            'C%C3%B4te+d%C2%B4Ivoire': 'CIV',
            'Ivory+Coast': 'CIV',
            'Channel+Islands': 'GBR'
            }


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = models.alexnet(pretrained=True).to(device)
    new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = new_classifier

    with_country = dataloader.dataset.with_country

    country_with_langs = {}
    country_with_imgs = {} # for each country, first list is tourist second is local
    lang_counts = {}

    detecter = fasttext.load_model('util_files/lid.176.bin')
    lang_dict = {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for i, (data, target) in enumerate(tqdm(dataloader)):
        if data is None:
            continue
        this_tags = [tag['label'] for tag in target[0] if len(tag['label']) >= 3]
        if len(this_tags) > 0:
            srcz = []
            conf = []
            for tag in this_tags:
                classify = detecter.predict(tag)
                srcz.append(classify[0][0][9:])
                conf.append(classify[1][0])

            # Pick out the most common language
            commons = Counter(srcz).most_common()
            the_src = commons[0][0]
            # If the most common language is English, look at the second most common language
            # since people oftentimes use English even when it's not their native language
            if the_src == 'en' and len(commons) > 1:
                the_src_maybe = commons[1][0]
                words = [i for i in range(len(srcz)) if srcz[i] == the_src_maybe]
                # If this second most common language has been classified with more than .5
                # probability, then choose this language for the image
                for word in words:
                    if conf[word] > .5: 
                        the_src = the_src_maybe
            if the_src in lang_counts.keys():
                lang_counts[the_src] += 1
            else:
                lang_counts[the_src] = 1

            country = target[2][0]
            iso3 = None
            local = None
            try:
                iso3 = pycountry.countries.search_fuzzy(country.replace('+', ' '))[0].alpha_3
            except LookupError:
                iso3 = missing[country]
            try:
                country_info = CountryInfo(country.replace('+', ' ')).info()
            except KeyError:
                country_info = {}
            country_name = country.split('+')
            if 'name' in country_info.keys():
                country_name += country_info['name']
            if 'nativeName' in country_info.keys():
                country_name += country_info['nativeName']

            # When comparing images to distinguish between tourist and local, we further look into the content of the tags,
            # allowing some images to be categorized as 'unknown' if we are not that sure if it's tourist or local

            # Local: in a local language, country's name isn't a tag, and 'travel' isn't a tag
            # Tourist: in a non-local language, or 'travel' is a tag
            try:
                if the_src in iso3_to_lang[iso3] and len(set(country_name)&set(this_tags)) == 0 and 'travel' not in this_tags:
                    local = 1
                elif the_src not in iso3_to_lang[iso3] or 'travel' in this_tags:
                    local = 0
            except KeyError:
                 print("This iso3 can't be found in iso3_to_lang: {}".format(iso3))

            if country not in country_with_langs.keys():
                country_with_langs[country] = []
                country_with_imgs[country] = [[], []]
            country_with_langs[country].append(the_src)
            if local is not None:
                if len(country_with_imgs[country][local]) < 500:
                    data = normalize(data).to(device)
                    big_data = F.interpolate(data.unsqueeze(0), size=224, mode='bilinear').to(device)
                    this_features = model.forward(big_data)
                    country_with_imgs[country][local].append((this_features.data.cpu().numpy(), target[3]))


    info = {}
    info['lang_counts'] = lang_counts
    info['country_with_langs'] = country_with_langs
    info['country_with_imgs'] = country_with_imgs

    pickle.dump(info, open("results/{}/geo_lng.pkl".format(args.folder), "wb"))

def geo_att(dataloader, args):
    geo_att = {}

    # code adapted from geo_ctr_gps
    geo_boundaries = dataloader.dataset.geo_boundaries
    geo_boundaries_key_name = dataloader.dataset.geo_boundaries_key_name

    # import subregion boundaries shapefile
    subregion_boundaries = dataloader.dataset.subregion_boundaries
    subregion_boundaries_key_name = dataloader.dataset.subregion_boundaries_key_name

    # fn that returns name of political region that a point falls into (eg. Manhattan)
    def bin_point(lng, lat, is_subregion):
        point = Point(lng, lat)
        # check each polygon to see if it contains the point
        if not is_subregion:
            for feature in geo_boundaries['features']:
                polygon = shape(feature['geometry'])
                if polygon.contains(point):
                    return feature['properties'][geo_boundaries_key_name]
            return None
        else:
            # subregion binning
            for feature in subregion_boundaries['features']:
                polygon = shape(feature['geometry'])
                if polygon.contains(point):
                    return feature['properties'][subregion_boundaries_key_name]
            return None

    for i, (data, target) in enumerate(tqdm(dataloader)):
        attribute = target[1]
        lat_lng = target[5]
        if len(attribute) > 1 and lat_lng is not None:
            region_name = (bin_point(float(lat_lng['lng']), float(lat_lng['lat']), False))   
            if subregion_boundaries is not None:
                subregion_names = bin_point(float(lat_lng['lng']), float(lat_lng['lat']), True)
            # add the subregion and region binnings
            for att in attribute[0]:
                if att not in geo_att:
                    if subregion_boundaries is not None: 
                        geo_att[att] = {'lat_lng': [], 'region': [], 'subregion': []}
                    else:
                        geo_att[att] = {'lat_lng': [], 'region': []}
                geo_att[att]['lat_lng'].append(lat_lng)
                geo_att[att]['region'].append(region_name)
                if subregion_boundaries is not None:
                    geo_att[att]['subregion'].append(subregion_names)
    pickle.dump(geo_att, open("results/{}/geo_att.pkl".format(args.folder), "wb"))
    
