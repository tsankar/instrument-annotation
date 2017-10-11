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

synsets = ['n03467517']
for elm in synsets:
    path = os.path.join(PATH_PREF, elm + '.tar')

    print 'Downloading ' + elm + '...'
    url = 'http://www.image-net.org/download/synset?wnid=' + elm + '&username=tharuns&accesskey=138012ac3e926a017d5103e6aeb38f603e8286d1&release=latest&src=stanford'
    urllib.urlretrieve(url, path)

    print 'Extracting...'

    tar = tarfile.open(path)
    new_dir = os.path.join(PATH_PREF, elm)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        tar.extractall(path=new_dir)

    tar.close()
