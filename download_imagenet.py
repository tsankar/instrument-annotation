'''
Downloads images from the relevant ImageNet synsets and extracts them

Usage:
python download_imagenet.py [path to save files (optional)]
'''
import urllib
import sys
import tarfile
import os
import shutil

def download_imagenet(PATH_PREF=''):
    os.chdir(PATH_PREF)

    IMAGE_PATH = os.path.join(PATH_PREF, 'Images')
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)
    ANNO_PATH = os.path.join(PATH_PREF, 'Annotations')
    if not os.path.exists(ANNO_PATH):
        os.makedirs(ANNO_PATH)

    synsets = ['n03467517', 'n02804123', 'n02676566', 'n03272010', 'n03499907', 'n03249569', 'n04249415']
    for elm in synsets:
        im_path = os.path.join(IMAGE_PATH, elm + '.tar')

        print 'Downloading images for ' + elm + '...'
        url = 'http://www.image-net.org/download/synset?wnid=' + elm + '&username=tharuns&accesskey=138012ac3e926a017d5103e6aeb38f603e8286d1&release=latest&src=stanford'
        urllib.urlretrieve(url, im_path)

        print 'Extracting...'

        tar = tarfile.open(im_path)
        new_dir = os.path.join(IMAGE_PATH, elm)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            tar.extractall(path=new_dir)

        tar.close()

        anno_path = os.path.join(ANNO_PATH, elm + '.tar.gz')
        # print 'Downloading Bounding Boxes for ' + elm + '...'
        # url = 'http://image-net.org/api/download/imagenet.bbox.synset?wnid=' + elm
        try:
            # urllib.urlretrieve(url, anno_path)
            print 'Extracting...'

            tar = tarfile.open(anno_path)
            new_dir = os.path.join(ANNO_PATH, elm)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                tar.extractall(path=new_dir)
        except IOError:
            print 'No bounding boxes for this synset'

        tar.close()

def move_images(PATH_PREF='', dest='/home/tharun/data/ILSVRC'):
    IMAGE_PATH = os.path.join(PATH_PREF, 'Images')
    ANNO_PATH = os.path.join(PATH_PREF, 'Annotations')

    dest_impath = os.path.join(dest, 'Data/DET/train')
    print 'Moving images...'
    dirs = [d for d in os.listdir(IMAGE_PATH) if os.path.isdir(os.path.join(IMAGE_PATH, d))]
    for folder in dirs:
        for im in os.listdir(os.path.join(IMAGE_PATH, folder)):
            shutil.copy(os.path.join(IMAGE_PATH, folder, im), dest_impath)

    dest_annopath = os.path.join(dest, 'Annotations/DET/train')
    print 'Moving BBs...'
    dirs = [d for d in os.listdir(ANNO_PATH) if os.path.isdir(os.path.join(ANNO_PATH, d))]
    for folder in dirs:
        for im in os.listdir(os.path.join(ANNO_PATH, folder, 'Annotation', folder)):
            shutil.copy(os.path.join(ANNO_PATH, folder, 'Annotation', folder, im), dest_annopath)
