import os
assert 'measurement' not in os.getcwd() and 'analysis_notebooks' not in os.getcwd(), "Script must be run from home directory"
import sys
sys.path.append('.')
import datasets
import torchvision.transforms as transforms
import pycountry
from scipy import stats
from sklearn import svm
import time
import pickle
import random
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import sqrt
import operator
import copy
import argparse
from sklearn.model_selection import permutation_test_score

# Projecting a set of features into a lower-dimensional subspace with PCA
def project(features, dim):
    standardized = StandardScaler().fit_transform(features)
    pca = PCA(n_components=dim)
    principalComponents = pca.fit_transform(X=standardized)
    return principalComponents

# Calculating the binomial proportion confidence interval
def wilson(p, n, z = 1.96):
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (lower_bound, upper_bound)

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

def sixprep(dataset, folder_name):
    if (dataset.geography_info_type == "STRING_FORMATTED_LABEL" and dataset.geography_label_string_type == "COUNTRY_LABEL"):
        info_stats = pickle.load(open("results/{}/geo_tag.pkl".format(folder_name), "rb")) #20GB
        country_tags = info_stats['country_tags']
        tag_to_subregion_features = info_stats['tag_to_subregion_features']
        iso3_to_subregion = pickle.load(open('iso3_to_subregion_mappings.pkl', 'rb'))
        categories = dataset.categories
        total_counts = np.zeros(len(categories))
        subregion_tags = {}
        for country, counts in country_tags.items():
            total_counts = np.add(total_counts, counts)
            subregion = iso3_to_subregion[country_to_iso3(country)]
            if subregion not in subregion_tags.keys():
                subregion_tags[subregion] = np.zeros(len(categories))
            subregion_tags[subregion] = np.add(subregion_tags[subregion], counts)
        total_counts = total_counts.astype(int)
        sum_total_counts = int(np.sum(total_counts))
        pvalues_over = {} # pvalue : '[country]: [tag] (country num and total num info for now)'
        pvalues_under = {} 
        if not os.path.exists("checkpoints/{}/geo_tag_a.pkl".format(folder_name)):
            for country, counts in country_tags.items():
                tags_for_country = int(np.sum(counts))
                if tags_for_country < 50: # threshold for country to have at least 50 tags so there are enough samples for analysis
                    continue
                for i, count in enumerate(counts):
                    this_counts = np.zeros(tags_for_country)
                    this_counts[:int(count)] = 1
                    that_counts = np.zeros(sum_total_counts - tags_for_country)
                    that_counts[:total_counts[i] - int(count)] = 1
                    p = stats.ttest_ind(this_counts, that_counts)[1]
                    tag_info = '{0}-{1} ({2}/{3} vs {4}/{5})'.format(country, categories[i], int(count), tags_for_country, int(total_counts[i] - count), sum_total_counts - tags_for_country)
                    if np.mean(this_counts) > np.mean(that_counts):
                        pvalues_over[p] = tag_info
                    else:
                        pvalues_under[p] = tag_info
            pickle.dump([pvalues_under, pvalues_over], open('checkpoints/{}/geo_tag_a.pkl'.format(folder_name), 'wb'))
        else:
            pvalues_under, pvalues_over = pickle.load(open('checkpoints/{}/geo_tag_a.pkl'.format(folder_name), 'rb'))

        import warnings
        warnings.filterwarnings("ignore")

        if not os.path.exists('checkpoints/{}/geo_tag_b.pkl'.format(folder_name)):
            phrase_to_value = {}
            ## Look at appearance differences in how a tag is represented across subregions
            for tag in tag_to_subregion_features.keys():
                subregion_features = tag_to_subregion_features[tag]
                all_subregions = list(subregion_features.keys())
                all_features = []
                all_filepaths = []
                start = 0
                for subregion in all_subregions:
                    this_features = [features[0] for features in subregion_features[subregion]]
                    this_filepaths = [features[1] for features in subregion_features[subregion]]
                    if len(this_features) > 0:
                        all_features.append(np.array(this_features)[:, 0, :])
                        all_filepaths.append(this_filepaths)
                if len(all_features) == 0:
                    continue
                all_features = np.concatenate(all_features, axis=0)
                all_filepaths = np.concatenate(all_filepaths, axis=0)
                labels = np.zeros(len(all_features))
                for j, subregion in enumerate(all_subregions):
                    labels[start:len(subregion_features[subregion])+start] = j
                    start += len(subregion_features[subregion])
                num_features = int(np.sqrt(len(all_features)))
                all_features = project(all_features, num_features)

                clf = svm.SVC(kernel='linear', probability=True, decision_function_shape='ovr', class_weight='balanced', max_iter=5000)
                clf_ovo = svm.SVC(kernel='linear', probability=False, decision_function_shape='ovo', class_weight='balanced')

                if len(np.unique(labels)) <= 1:
                    continue
                clf.fit(all_features, labels)
                clf_ovo.fit(all_features, labels)
                acc = clf.score(all_features, labels)
                acc_ovo = clf_ovo.score(all_features, labels)
                probs = clf.decision_function(all_features)

                class_preds = clf.predict(all_features)
                class_probs = clf.predict_proba(all_features)

                j_to_acc = {}
                for j, subregion in enumerate(all_subregions):
                    if j in labels:
                        # to get acc in subregion vs out
                        this_labels = np.copy(labels)
                        this_labels[np.where(labels!=j)[0]] = -1
                        this_preds = np.copy(class_preds)
                        this_preds[np.where(class_preds!=j)[0]] = -1
                        this_acc = np.mean(this_preds == this_labels)
                        j_to_acc[j] = this_acc

                        # different version of accuracy
                        # indices = np.where(labels == j)[0]
                        # this_acc = np.mean(labels[indices] == class_preds[indices])
                        # wilson_acc = wilson(this_acc, len(indices))[0]
                        # j_to_acc[j] = wilson_acc # so that country with one image isn't most accurate
                        # #j_to_acc[j] = this_acc

                fig = plt.figure(figsize=(16, 12))
                plt.subplots_adjust(hspace=.48)
                fontsize = 24
                diff_subregion = max(j_to_acc.items(), key=operator.itemgetter(1))[0]
                subregion_index = list(clf.classes_).index(diff_subregion)
                class_probs = class_probs[:, subregion_index]
                in_sub = np.where(labels == diff_subregion)[0]
                out_sub = np.where(labels != diff_subregion)[0]
                in_probs = class_probs[in_sub]
                out_probs = class_probs[out_sub]
                in_indices = np.argsort(in_probs)
                out_indices = np.argsort(out_probs)

                original_labels = np.copy(labels)
                def subregion_scoring(estimator, X_test, y_test):
                    y_pred = estimator.predict(X_test)
                    y_test[np.where(y_test!=diff_subregion)[0]] = -1
                    y_pred[np.where(y_pred!=diff_subregion)[0]] = -1
                    acc_random = np.mean(y_test == y_pred)
                    return acc_random

                base_acc, rand_acc, p_value = permutation_test_score(clf, all_features, labels, scoring=subregion_scoring, n_permutations=100)
                value = base_acc/np.mean(rand_acc)
                if p_value > .05 and value < 1.2: # can tune as desired
                    continue

                phrase = dataset.labels_to_names[dataset.categories[tag]]
                phrase_to_value[phrase] = [value, all_subregions[diff_subregion], acc, p_value, num_features, j_to_acc]
                
                pickle.dump([original_labels, class_probs, class_preds, diff_subregion, all_filepaths], open('results/{0}/{1}/{2}_info.pkl'.format(folder_name, 'geo_tag', dataset.labels_to_names[dataset.categories[tag]]), 'wb'))
            pickle.dump(phrase_to_value, open('checkpoints/{}/geo_tag_b.pkl'.format(folder_name), 'wb'))
        else:
            phrase_to_value = pickle.load(open('checkpoints/{}/geo_tag_b.pkl'.format(folder_name), 'rb'))
            
    elif (dataset.geography_info_type == "STRING_FORMATTED_LABEL" and dataset.geography_label_string_type == "REGION_LABEL"):
        info_stats = pickle.load(open("results/{}/geo_tag.pkl".format(folder_name), "rb")) 
        region_tags = info_stats['region_tags']
        tag_to_region_features = info_stats['tag_to_region_features']

        categories = dataset.categories
        total_counts = np.zeros(len(categories))

        for region, counts in region_tags.items():
            total_counts = np.add(total_counts, counts)

        total_counts = total_counts.astype(int)
        sum_total_counts = int(np.sum(total_counts))

        if not os.path.exists('checkpoints/{}/geo_tag_a.pkl'.format(folder_name)):
            pvalues_over = {} # pvalue : '[region]: [tag] (region num and total num info for now)'
            pvalues_under = {} 
            for region, counts in region_tags.items():
                tags_for_region = int(np.sum(counts))
                if tags_for_region < 50: # threshold for region to have at least 50 tags so there are enough samples for analysis
                    continue
                for i, count in enumerate(counts):
                    this_counts = np.zeros(tags_for_region)
                    this_counts[:int(count)] = 1
                    that_counts = np.zeros(sum_total_counts - tags_for_region)
                    that_counts[:total_counts[i] - int(count)] = 1
                    p = stats.ttest_ind(this_counts, that_counts)[1]
                    tag_info = '{0}-{1} ({2}/{3} vs {4}/{5})'.format(region, categories[i], int(count), tags_for_region, int(total_counts[i] - count), sum_total_counts - tags_for_region)
                    if np.mean(this_counts) > np.mean(that_counts):
                        pvalues_over[p] = tag_info
                    else:
                        pvalues_under[p] = tag_info
            pickle.dump([pvalues_under, pvalues_over], open('checkpoints/{}/geo_tag_a.pkl'.format(folder_name), 'wb'))
        else:
            pvalues_under, pvalues_over = pickle.load(open('checkpoints/{}/geo_tag_a.pkl'.format(folder_name), 'rb'))

    elif dataset.geography_info_type == "GPS_LABEL":
        info_stats = pickle.load(open("results/{}/geo_tag.pkl".format(folder_name), "rb")) 
        region_tags = info_stats['region_tags']
        subregion_tags = info_stats.get('subregion_tags', None)
        tag_to_region_features = info_stats['tag_to_region_features']

        categories = dataset.categories
        total_counts = np.zeros(len(categories))

        for region, counts in region_tags.items():
            total_counts = np.add(total_counts, counts)

        total_counts = total_counts.astype(int)
        sum_total_counts = int(np.sum(total_counts))

        if not os.path.exists('checkpoints/{}/geo_tag_a.pkl'.format(folder_name)):
            pvalues_over = {} # pvalue : '[region]: [tag] (region num and total num info for now)'
            pvalues_under = {} 
            for region, counts in region_tags.items():
                tags_for_region = int(np.sum(counts))
                if tags_for_region < 50: # threshold for region to have at least 50 tags so there are enough samples for analysis
                    continue
                for i, count in enumerate(counts):
                    this_counts = np.zeros(tags_for_region)
                    this_counts[:int(count)] = 1
                    that_counts = np.zeros(sum_total_counts - tags_for_region)
                    that_counts[:total_counts[i] - int(count)] = 1
                    p = stats.ttest_ind(this_counts, that_counts)[1]
                    tag_info = '{0}-{1} ({2}/{3} vs {4}/{5})'.format(region, categories[i], int(count), tags_for_region, int(total_counts[i] - count), sum_total_counts - tags_for_region)
                    if np.mean(this_counts) > np.mean(that_counts):
                        pvalues_over[p] = tag_info
                    else:
                        pvalues_under[p] = tag_info
            pickle.dump([pvalues_under, pvalues_over], open('checkpoints/{}/geo_tag_a.pkl'.format(folder_name), 'wb'))
        else:
            pvalues_under, pvalues_over = pickle.load(open('checkpoints/{}/geo_tag_a.pkl'.format(folder_name), 'rb'))

def tenprep(dataset, folder_name):
    iso3_to_subregion = pickle.load(open('iso3_to_subregion_mappings.pkl', 'rb'))
    mappings = pickle.load(open('util_files/country_lang_mappings.pkl', 'rb'))
    iso3_to_lang = mappings['iso3_to_lang']
    lang_to_iso3 = mappings['lang_to_iso3']

    lang_info = pickle.load(open('results/{}/geo_lng.pkl'.format(folder_name), 'rb'))
    counts = lang_info['lang_counts']
    country_with_langs = lang_info['country_with_langs']
    country_with_imgs = lang_info['country_with_imgs']

    to_write_lower = {}
    to_write_upper = {}
    iso3_to_percent = {}
    subregion_to_percents = {}
    subregion_to_filepaths = {} # 0 is tourist, 1 is local
    subregion_to_embeddings = {} # 0 is tourist, 1 is local
    for country in country_with_langs.keys():
        iso3 = country_to_iso3(country)
        langs_in = 0
        langs_out = {}
        for lang in country_with_langs[country]:
            try:
                if lang in iso3_to_lang[iso3]:
                    langs_in += 1
                else:
                    if lang in langs_out.keys():
                        langs_out[lang] += 1
                    else:
                        langs_out[lang] = 1
            except KeyError:
                 print("This iso3 can't be found in iso3_to_lang: {}".format(iso3))
        this_total = len(country_with_langs[country])
        others = ''
        for lang in langs_out.keys():
            if len(lang) == 2:
                lang_name = pycountry.languages.get(alpha_2=lang)
            elif len(lang) == 3:
                lang_name = pycountry.languages.get(alpha_3=lang)
            else:
                print("{} is not 2 or 3 letters?".format(lang))
            if lang_name is not None:
                lang_name = lang_name.name
            else:
                lang_name = lang
            others += lang_name + ": " + str(round(langs_out[lang]/this_total, 4)) + ", "
        if iso3 is not None:
            subregion = iso3_to_subregion[iso3]
            if subregion in subregion_to_percents.keys():
                subregion_to_percents[subregion][0] += langs_in
                subregion_to_percents[subregion][1] += this_total
                subregion_to_filepaths[subregion][0].extend([chunk[1] for chunk in country_with_imgs[country][0]])
                subregion_to_filepaths[subregion][1].extend([chunk[1] for chunk in country_with_imgs[country][1]])
                subregion_to_embeddings[subregion][0].extend([chunk[0] for chunk in country_with_imgs[country][0]])
                subregion_to_embeddings[subregion][1].extend([chunk[0] for chunk in country_with_imgs[country][1]])
            else:
                subregion_to_percents[subregion] = [langs_in, this_total]
                subregion_to_filepaths[subregion] = [[chunk[1] for chunk in country_with_imgs[country][0]], [chunk[1] for chunk in country_with_imgs[country][1]]]
                subregion_to_embeddings[subregion] = [[chunk[0] for chunk in country_with_imgs[country][0]], [chunk[0] for chunk in country_with_imgs[country][1]]]
        #local_percent = langs_in / this_total
        tourist_percent = 1.0 - (langs_in / this_total)
        lp_under, lp_over = wilson(tourist_percent, this_total)
        phrase = '{0} has {1}% non-local tags, and the extra tags are:\n\n{2}'.format(country, round(100.*tourist_percent, 4), others)
    #     to_write_lower[lp_under] = phrase
    #     to_write_upper[lp_over] = phrase
        to_write_lower[country] = [phrase, tourist_percent]
        #iso3_to_percent[iso3] = local_percent
        iso3_to_percent[iso3] = lp_under

    subregion_to_accuracy = {}
    subregion_to_percents_phrase = {}

    for key in subregion_to_percents.keys():
        if not os.path.exists('results/{0}/{1}/{2}_info.pkl'.format(folder_name, 'geo_lng', key.replace(' ', '_'))):
            low_bound, high_bound = wilson(1 - subregion_to_percents[key][0] / subregion_to_percents[key][1], subregion_to_percents[key][1])

            clf = svm.SVC(kernel='linear', probability=False, decision_function_shape='ovr', class_weight='balanced')
            clf_random = svm.SVC(kernel='linear', probability=False, decision_function_shape='ovr', class_weight='balanced')
            tourist_features = subregion_to_embeddings[key][0]
            local_features = subregion_to_embeddings[key][1]
            if len(tourist_features) == 0 or len(local_features) == 0:
                continue
            tourist_features, local_features = np.array(tourist_features)[:, 0, :], np.array(local_features)[:, 0, :]
            all_features = np.concatenate([tourist_features, local_features], axis=0)
            num_features = int(np.sqrt(len(all_features)))
            all_features = project(all_features, num_features)
            labels = np.zeros(len(all_features))
            labels[len(tourist_features):] = 1
            clf.fit(all_features, labels)
            acc = clf.score(all_features, labels)
            probs = clf.decision_function(all_features)

            np.random.shuffle(all_features)
            clf_random.fit(all_features, labels)
            acc_random = clf_random.score(all_features, labels)
            value = acc / acc_random

            subregion_to_percents_phrase[key] = [subregion_to_percents[key][0] / subregion_to_percents[key][1], '[{0} - {1}] for {2}'.format(round(low_bound, 4), round(high_bound, 4), subregion_to_percents[key][1])]
            subregion_to_accuracy[key] = [acc, value, len(tourist_features), len(all_features), num_features]
            tourist_probs = []
            local_probs = []
            for j in range(len(all_features)):
                if j < len(tourist_features):
                    tourist_probs.append(-probs[j])
                else:
                    local_probs.append(probs[j])        
            pickle.dump([labels, tourist_probs, local_probs, subregion_to_filepaths[key]], open('results/{0}/{1}/{2}_info.pkl'.format(folder_name, 'geo_lng', key.replace(' ', '_')), 'wb'))
    subregion_local_svm_loc = 'results/{0}/{1}/subregion_svm.pkl'.format(folder_name, 'geo_lng')
    if not os.path.exists(subregion_local_svm_loc):
        pickle.dump([subregion_to_accuracy, subregion_to_percents_phrase], open(subregion_local_svm_loc, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prerun for gender')
    parser.add_argument('--dataset', type=str, default='openimages',
            help='input dataset to format')
    parser.add_argument('--folder', type=str, default='random',
            help='folder to store results in')

    args = parser.parse_args()

    transform_train = transforms.Compose([ 
                transforms.ToTensor()
                        ])


    if not os.path.exists("checkpoints/{}".format(args.folder)):
        os.mkdirs("checkpoints/{}".format(args.folder), exist_ok=True)

    if args.dataset == 'openimages':
        dataset = datasets.OpenImagesDataset(transform_train)
    elif args.dataset == 'coco':
        dataset = datasets.CoCoDataset(transform_train)
    elif args.dataset == 'sun':
        dataset = datasets.SUNDataset(transform_train)
    elif args.dataset == 'imagenet':
        dataset = datasets.ImagenetDataset(transform_train)
    elif args.dataset == 'yfcc':
        dataset = datasets.YfccPlacesDataset(transform_train, 'geo_tag')
    elif args.dataset == 'cityscapes':
        dataset = datasets.CityScapesDataset(transform_train)

    if (not os.path.exists("results/{}/geo_tag.pkl".format(args.folder))) and (not os.path.exists("results/{}/geo_tag.pkl".format(args.folder))) and (not os.path.exists("results/{}/geo_tag.pkl".format(args.folder))):
        print("geo_tag Metric was not run for this dataset.")
    else:
        sixprep(dataset, args.folder)
    
    if args.dataset == 'yfcc':
        dataset = datasets.YfccPlacesDataset(transform_train, 'geo_lng')

    if not os.path.exists("results/{}/geo_lng.pkl".format(args.folder)):
        print("geo_lng Metric was not run for this dataset.")
        exit()
    tenprep(dataset, args.folder)



