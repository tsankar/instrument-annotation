'''
Downloads images from the relevant ImageNet synsets and extracts them

Usage:
python download_imagenet.py [path to save files (optional)]
'''
import urllib
import sys
import tarfile
import os

PATH_PREF = ''

if(len(sys.argv) > 1):
    PATH_PREF = sys.argv[1]

os.chdir(PATH_PREF)

IMAGE_PATH = os.path.join(PATH_PREF, 'Images')
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)
ANNO_PATH = os.path.join(PATH_PREF, 'Annotations')
if not os.path.exists(ANNO_PATH):
    os.makedirs(ANNO_PATH)

synsets = ['n03467517']
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

    anno_path = os.path.join(ANNO_PATH, elm + '.tar')
    print 'Downloading Bounding Boxes for ' + elm + '...'
    url = 'http://image-net.org/api/download/imagenet.bbox.synset?wnid=' + elm
    try:
        urllib.urlretrieve(url, anno_path)
        print 'Extracting...'

        tar = tarfile.open(anno_path)
        new_dir = os.path.join(ANNO_PATH, elm)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            tar.extractall(path=new_dir)
    except IOError:
        print 'No bounding boxes for this synset'

    tar.close()
