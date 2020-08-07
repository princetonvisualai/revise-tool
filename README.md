# REVISE: REvealing VIsual biaSEs
A tool that automatically detects possible forms of bias in a visual dataset along the axes of object-based, gender-based, and geography-based patterns, and from which next steps for mitigation are suggested. 

In the *sample_summary_pdfs* folder there are examples of the kinds of auto-generated summaries our tool outputs along each axis for a dataset. These samples are annotated in orange with some notes on how to interpret them.

## Table of Contents

* [Setup](https://github.com/princetonvisualai/revise-tool#setup)
* [Steps to perform analysis](https://github.com/princetonvisualai/revise-tool#steps-to-perform-analysis)
* [Measurements](https://github.com/princetonvisualai/revise-tool#measurements)
* [Potential Environment Issues](https://github.com/princetonvisualai/revise-tool#potential-environment-issues)
* [Paper and Citation](https://github.com/princetonvisualai/revise-tool#paper-and-citation)

## Setup:
- Clone this repo
- Set up the conda environment using the appropriate yml file (information in [Potential Environment Issues](https://github.com/princetonvisualai/revise-tool#potential-environment-issues))
```
conda env create -f environments/[environment].yml
```
- Download the models with
```
bash download.sh
```

- Note: we use [Amazon Rekognition](https://aws.amazon.com/rekognition/)'s proprietery facial detection tool in our analyses, which does incur a charge, and this will need to be set up for each user (instructions on Amazon's site). There are many free facial detection tools available as well, and you can change what is used in gender_based.py . One such free facial detection tool through cv2 is already implemented, and simply involves changing the FACE_DETECT variable in gender_based.py from 0 to 1 to use this instead.

## Steps to perform analysis:
(0.5 optional) To experiment with the tool on the Coco dataset for Object-Based and Gender-Based metrics without having to run all the measurements on a dataset first, download the pickle files from [here](https://drive.google.com/drive/folders/1cGUr2ruV7IRl4h8EGtCjRCsg8wtPVu5P?usp=sharing), and place them in a folder in the tool directory called results/[folder name], then skip to Step 3.

(1) Make a dataloader structured like the 'Template Dataset' in datasets.py (add to main_measure.py as well), and fill in with the dataset you would like to analyze

(2) Run main_measure to make a pass through the data and collect the metrics for analysis, for example to get measurements 1, 2, 3, 4, 9, 10, 11 on COCO and have the file be saved in coco_example:
```
python3 main_measure.py --measurements 1 2 3 4 9 10 11 --dataset 'coco' --folder 'coco_example'
```

(2.5 optional) 
To optionally do some of the processing ahead of time so interacting with the notebook can be faster, for the Gender notebook (M4) run
```
python3 prerun_analyzegen.py --dataset 'coco' --folder 'coco_example'
```
and for the Geography notebook (M6 and M10) run
```
python3 prerun_analyzegeo.py --dataset 'yfcc' --folder 'yfcc_example'
```

(3) Open up the jupyter notebook corresponding to the axis of bias you would like to explore: object, gender, or geography. Further instructions are at the top of the notebook about how to run them.

## Measurements
Measurements that can be run, along with the file and name of the function they are associated with:

### Object-Based
(Note: M0, M7, M8 actually all run the same function, so for main_measure.py it's only necessary to run one of these to get all the measurements)

M0: object_based.count_cooccurrence\
Counts the number of times each instance occurs, coocurrence of instances occurs, and supercateogry occurs

M7: object_based.supercategory_size_and_distance\
Counts the size and distance from center at the supercategory level

M8: object_based.supercategory_with_people\
Counts how much supercategories are represented with or without people

M9: object_based.scene_categorization\
Counts overall scenes, scene-supercategory cooccurrences, scene-instance cooccurrences, and gets features per scene per supercategory

### Gender-Based

M1: gender_based.size_and_distance\
Gets the size of the person and distance from center, as well as if a face is detected

M2: gender_based.count_cooccurrence\
Counts how often each gender occurs with an instance and instance pair

M3: gender_based.distance_for_instance\
Calculates the distance each gender is from each object

M4: gender_based.cluster_for_instance\
Gets scene-level and cropped object-level features per object class for each gender

M11: gender_based.scenes\
Counts the types of scenes each gender occurs with

### Geography-Based

M5: geography_based.count_country\
Counts the number of images from each country

M6: geography_based.count_tags\
Counts the number of tags from each country, as well as extracts AlexNet features pretrained on ImageNet for each tag, grouping by subregion

M10: geography_based.count_langs\
Counts the languages that make up the image tags, and whether or not they are local to the country the image is from. Also extracts image-level features to compare if locals and tourist portray a country differently

## Potential Environment Issues
- If FileNotFoundError: [Errno 2] No such file or directory: appears from importing basemap at epsgf = open(os.path.join(pyproj_datadir,'epsg')), change the PROJ_LIB variable as suggested [here](https://stackoverflow.com/questions/58683341/basemap-wont-import-because-epsg-file-or-directory-cant-be-found-macos-ana).
In the jupyter notebook, this may involve setting it in a cell like
```
import os
os.environ['PROJ_LIB'] = '/new/folder/location/of/epsg'
```
If the epsg file is still not found, it can be downloaded manually from [here](https://raw.githubusercontent.com/matplotlib/basemap/master/lib/mpl_toolkits/basemap/data/epsg), with the path locaation set as mentioned.
- For MacOS, use environments/environment_mac.yml, and if there are errors, try running the following commands first
```
conda config --set allow_conda_downgrades true
conda install conda=4.6.14
```
- environments/environment.yml is for linux machines, try environments/environment_nobuild.yml for other OS's
- Try deleting line 9 of environments/enivronment.yml of ```_libgcc_mutex=0.1=main``` if there are compatability errors

## Paper and Citation
[REVISE: A Tool for Measuring and Mitigating Bias in Image Datasets](https://arxiv.org/abs/2004.07999)

```
@article{revisetool,
Author = {Angelina Wang and Arvind Narayanan and Olga Russakovsky},
Title = {{REVISE}: A Tool for Measuring and Mitigating Bias in Image Datasets},
Year = {2020},
Journal = {European Conference on Computer Vision (ECCV)},
}
```

## Funding
This work is partially supported by the National Science Foundation under Grant No.
1763642 and No. 1704444.
