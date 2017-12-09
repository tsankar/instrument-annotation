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

def download_imagenet(PATH_PREF='/home/tharun/data'):
    os.chdir(PATH_PREF)

    IMAGE_PATH = os.path.join(PATH_PREF, 'Images')
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)
    ANNO_PATH = os.path.join(PATH_PREF, 'Annotations')
    if not os.path.exists(ANNO_PATH):
        os.makedirs(ANNO_PATH)

    synsets = ['n00007846', 'n03467517', 'n02804123', 'n02676566', 'n03272010', 'n03499907', 'n03249569', 'n04249415']
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

def move_images(PATH_PREF='/home/tharun/data', dest='/home/tharun/data/ILSVRC'):
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

def create_imgsets():
    ROOT_DIR = '/home/tharun/data/ILSVRC/ImageSets/DET'
    for d in ['train1']: # 'train', 'val', 'test'
        DATA_DIR = os.path.join('/home/tharun/data/ILSVRC/Annotations/DET/', d)

        f = open(os.path.join(ROOT_DIR, d + '.txt'), 'w')
        i = 1
        for im in os.listdir(DATA_DIR):
            f.write(im.strip('.xml') + ' ' + str(i) + '\n')
            i += 1

        f.close()

def split_data():
    ROOT = '/home/tharun/data/ILSVRC/Data/DET/'
    ims = os.listdir(os.path.join(ROOT, 'train'))
    num = len(ims)
    test_split = 0.1 * num
    val_split = 0.3 * num
    i = 0
    while i < test_split:
        im = ims[i]
        shutil.move(os.path.join(ROOT, 'train', im), os.path.join(ROOT, 'test', im))
        i += 1

    while i < val_split:
        im = ims[i]
        shutil.move(os.path.join(ROOT, 'train', im), os.path.join(ROOT, 'val', im))
        i += 1

    split_annos()

def split_annos():
    ROOT = '/home/tharun/data/ILSVRC/Annotations/DET'
    IM_ROOT = '/home/tharun/data/ILSVRC/Data/DET/'
    annos = os.listdir(os.path.join(ROOT, 'train'))

    for anno in annos:
        name = anno.split('.')[0]
        if os.path.exists(os.path.join(IM_ROOT, 'val', name + '.JPEG')):
            shutil.move(os.path.join(ROOT, 'train', anno), os.path.join(ROOT, 'val', anno))
        elif os.path.exists(os.path.join(IM_ROOT, 'test', name + '.JPEG')):
            shutil.move(os.path.join(ROOT, 'train', anno), os.path.join(ROOT, 'test', anno))


import xml.etree.ElementTree

def inspect_xmls():
    ANNO_DIR = '/home/tharun/data/ILSVRC/Annotations/DET/train'
    for f in os.listdir(ANNO_DIR):
        e = xml.etree.ElementTree.parse(os.path.join(ANNO_DIR, f)).getroot()
        for obj in e.findall('object'):
            print obj.find('name').text + '|'

import imgaug as ia
from imgaug import augmenters as iaa
import cv2

def augment():
    ia.seed(1)

    IMG_DIR = '/home/tharun/data/ILSVRC/Data/DET/train'
    ANNO_DIR = '/home/tharun/data/ILSVRC/Annotations/DET/train'
    im_save = '/home/tharun/data/ILSVRC/Data/DET/train1'
    anno_save = '/home/tharun/data/ILSVRC/Annotations/DET/train1'

    seq = iaa.Sequential([
        iaa.CropAndPad(percent=(-0.25, 0.25))
    ])

    for f in os.listdir(ANNO_DIR):
        name = f.split('.')[0]
        anno_path = os.path.join(ANNO_DIR, f)
        img_path = os.path.join(IMG_DIR, name + '.JPEG')

        if not os.path.exists(img_path):
            continue

        # Save original image/BB in new dir
        shutil.copy2(anno_path, anno_save)
        shutil.copy2(img_path, im_save)

        image = cv2.imread(img_path)

        tree = xml.etree.ElementTree.parse(anno_path)
        e = tree.getroot()
        bbox = e.find('object').find('bndbox')
        xmin = bbox.find('xmin')
        ymin = bbox.find('ymin')
        xmax = bbox.find('xmax')
        ymax = bbox.find('ymax')
        fname = e.find('filename')

        bbs = ia.BoundingBoxesOnImage([
            ia.BoundingBox(x1=int(xmin.text), y1=int(ymin.text), x2=int(xmax.text), y2=int(ymax.text))
        ], shape=image.shape)

        # Create and save 4 augmented versions of the image
        for i in range(0, 4):
            seq_det = seq.to_deterministic()

            image_aug = seq_det.augment_images([image])[0]
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            bbs_aug = bbs_aug.remove_out_of_image().cut_out_of_image()

            new_name = name + '_aug' + str(i)
            cv2.imwrite(os.path.join(im_save, new_name + '.JPEG'), image_aug)

            fname.text = new_name
            if bbs_aug.bounding_boxes:
                boxes = bbs_aug.bounding_boxes[0]
                xmin.text = str(boxes.x1)
                ymin.text = str(boxes.y1)
                xmax.text = str(boxes.x2)
                ymax.text = str(boxes.y2)
                tree.write(os.path.join(anno_save, new_name + '.xml'))
