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
    counts = {}

    for i, (data, target) in enumerate(tqdm(dataloader)):
        country = target[2][0]
        if country not in counts.keys():
            counts[country] = 0
        counts[country] += 1

    pickle.dump(counts, open("results/{}/geo_ctr.pkl".format(args.folder), "wb"))

def geo_tag(dataloader, args):
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


