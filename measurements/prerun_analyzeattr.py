import os
assert 'measurement' not in os.getcwd() and 'analysis_notebooks' not in os.getcwd(), "Script must be run from home directory"
import sys
sys.path.append('.')
import datasets
import torchvision.transforms as transforms
import pycountry
from scipy import stats
from sklearn import svm
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import sqrt
import operator
import copy
import argparse
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import permutation_test_score
import warnings
warnings.filterwarnings("ignore")

def main(dataset, folder_name):
    COLORS = sns.color_palette('Set2', 2)

    if not os.path.exists("checkpoints/{}".format(folder_name)):
        os.mkdirs("checkpoints/{}".format(folder_name), exist_ok=True)

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

    import warnings
    warnings.filterwarnings("ignore")

    if not os.path.exists("results/{0}/att_clu/".format(folder_name)):
        os.mkdir("results/{0}/att_clu/".format(folder_name))
    categories = dataset.categories
    names = dataset.labels_to_names
    stats_dict = pickle.load(open("results/{}/att_clu.pkl".format(folder_name), "rb"))
    instances = stats_dict['instance']
    scenes = stats_dict['scene']
    scene_filepaths = stats_dict['scene_filepaths']

    file_name = 'util_files/categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    scene_classes = tuple(classes)

    topn = 15

    plot_kwds = {'alpha' : .8, 's' : 30, 'linewidths':0}

    instance_p_values = []
    scene_p_values = []

    if not os.path.exists("checkpoints/{}/att_clu.pkl".format(folder_name)):
        value_to_phrase = {}
        value_to_scenephrase = {}
        for i in range(len(categories)):
            # SVM's to classify between an object's features for the genders
            clf = svm.SVC(kernel='linear', probability=False, max_iter=5000)
            clf_prob = svm.SVC(kernel='linear', probability=True)
            if len(instances[i][0]) <= 1 or len(instances[i][1]) <= 1 or len(scenes[i][0]) <= 1 or len(scenes[i][1]) <= 1:
                scene_p_values.append(float('inf'))
                instance_p_values.append(float('inf'))
                continue
            features_instances = np.concatenate([instances[i][0], instances[i][1]], axis=0)
            boundary_instances = len(instances[i][0])
            features_scenes = np.concatenate([scenes[i][0], scenes[i][1]], axis=0)
            boundary_scenes = len(scenes[i][0])

            ## Uncomment to visualize features of cropped object, saved as a png
            #projection_instances = TSNE().fit_transform(features_instances)
            #plt.scatter(*projection_instances.T, **plot_kwds, c=[COLORS[0] if i < boundary_instances else COLORS[1] for i in range(len(projection_instances))])
            #plt.savefig("results/{0}/{1}/instances_{2}.png".format(folder_name, att_clu, i))
            #plt.close()

            t, p = stats.ttest_ind(instances[i][0], instances[i][1])
            instance_p_values.append(np.nanmean(p))

            ## Uncomment to visualize features of entire scene, saved as a png
            #projection_scenes = TSNE().fit_transform(features_scenes)
            #plt.scatter(*projection_scenes.T, **plot_kwds, c=[COLORS[0] if i < boundary_scenes else COLORS[1] for i in range(len(projection_scenes))])
            #plt.savefig("results/{0}/{1}/scenes_{2}.png".format(folder_name, att_clu, i))
            #plt.close()

            t, p = stats.ttest_ind(scenes[i][0], scenes[i][1])
            scene_p_values.append(np.nanmean(p))
            num_features = int(np.sqrt(len(features_scenes)))

            labels = np.zeros(len(features_scenes))
            labels[len(scenes[i][0]):] = 1
            projected_features_scenes = StandardScaler().fit_transform(project(features_scenes, num_features))
            clf.fit(projected_features_scenes, labels)
            clf_prob.fit(projected_features_scenes, labels)
            acc = clf.score(projected_features_scenes, labels)
            probs = clf.decision_function(projected_features_scenes)
            scaled_probs = clf_prob.predict_proba(projected_features_scenes)
            a_probs = []
            b_probs = []
            preds = clf.predict(projected_features_scenes)
            scenes_per_gender = [[[], []] for i in range(len(scene_classes))]
            for j in range(len(features_scenes)):

                if j < len(scenes[i][0]):
                    a_probs.append(-probs[j])
                    this_scene = scene_filepaths[i][0][j][1]
                    scenes_per_gender[this_scene][0].append(np.absolute(scaled_probs[j][0]))
                else:
                    b_probs.append(probs[j])
                    this_scene = scene_filepaths[i][1][j - len(scenes[i][0])][1]
                    scenes_per_gender[this_scene][1].append(np.absolute(scaled_probs[j][1]))
            a_indices = np.argsort(np.array(a_probs))
            b_indices = np.argsort(np.array(b_probs))

            pickle.dump([a_indices, b_indices, scene_filepaths[i], a_probs, b_probs], open("results/{0}/att_clu/{1}_info.pkl".format(folder_name, names[categories[i]]), "wb"))

            base_acc, rand_acc, p_value = permutation_test_score(clf, projected_features_scenes, labels, scoring="accuracy", n_permutations=100)
            ratio = base_acc/np.mean(rand_acc)

            if p_value > 0.05 and ratio <= 1.2: # can tune as desired
                continue

            amount = len(features_instances)
            phrase = [ratio, names[categories[i]], acc, p_value, len(features_instances), num_features]
            value_to_phrase[i] = phrase

            for j in range(len(scene_classes)):
                a_dists = scenes_per_gender[j][0]
                b_dists = scenes_per_gender[j][1]
                a = np.zeros(len(scenes[i][0]))
                a[:len(a_dists)] = 1 #a_dists
                b = np.zeros(len(scenes[i][1]))
                b[:len(b_dists)] = 1 #b_dists
                _, p = stats.ttest_ind(a, b)
                if not np.isnan(p):
                    value_to_scenephrase[p] = [names[categories[i]], scene_classes[j], len(a_dists), len(a), len(b_dists), len(b)]
        pickle.dump([value_to_phrase, value_to_scenephrase], open("checkpoints/{}/att_clu.pkl".format(folder_name), 'wb'))
    else:
        value_to_phrase = pickle.load(open("checkpoints/{}/att_clu.pkl".format(folder_name), 'rb'))


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

    if not os.path.exists("results/{}/att_clu.pkl".format(args.folder)):
        print("att_clu Metric was not run for this dataset.")
        exit()

    if args.dataset == 'openimages':
        dataset = datasets.OpenImagesDataset(transform_train)
    elif args.dataset == 'coco':
        dataset = datasets.CoCoDataset(transform_train)
    elif args.dataset == 'sun':
        dataset = datasets.SUNDataset(transform_train)
    elif args.dataset == 'imagenet':
        dataset = datasets.ImagenetDataset(transform_train)

    main(dataset, args.folder)


