# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.utils.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
from PIL import Image


class vgg(imdb):
  def __init__(self, image_set, year):
    imdb.__init__(self, 'vgg_' + year + '_' + image_set)

    self.root = "/home/bruno/Projects/business/misc/datasets/flowcharts/scans"
    # name, paths
    self._year = year
    self._image_set = image_set

    self._classes = ('__background__', 'rectangle', 'paralellogram', 'diamond', 'oval')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

    self._image_data = self._parse_vgg_json()

    print(map(lambda d : d["filename"], self._image_data))
    # Default to roidb handler
    self.set_proposal_method('gt')

    self._roidb = self.gt_roidb()


    # Dataset splits that have ground-truth annotations (test splits
    # do not have gt annotations)
    self._gt_splits = ('train', 'val', 'minival')


  def _parse_vgg_json(self):
    """
    Parse the vgg into something sensible
    """
    data = {}
    with open(self.root + '/scan_fc_labelled.json', 'r') as outfile:  
        data = json.load(outfile)

    img_data = []

    for img in data["_via_img_metadata"]:
        img_data += [data["_via_img_metadata"][img]]

    return img_data


  def image_path_at(self, i):
    fasdf
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_id_at(self, i):
    asfs
    """
    Return the absolute path to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    afasdf
    """
    Construct an image path from the image's "index" identifier.
    """
    # Example image path for index=119993:
    #   images/train2014/COCO_train2014_000000119993.jpg
    file_name = ('COCO_' + self._data_name + '_' +
                 str(index).zfill(12) + '.jpg')
    image_path = osp.join(self._data_path, 'images',
                          self._data_name, file_name)
    assert osp.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_roi(self, data, idx):
    filename = data["filename"]
    print(filename)
    im = Image.open(self.root + '/' + filename)
    width, height = im.size

    # Extract all regions
    regions = data["regions"]

    num_objs = len(regions)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    for idx, region in enumerate(regions):
      shape = region["shape_attributes"]
      x, y, w, h = (int(shape["x"]), int(shape["y"]), int(shape["width"]), int(shape["height"]))
      boxes[idx] = [x, y, x + w, y + h]

      gt_classes[idx] = self._class_to_ind[region["region_attributes"]["shape"]]
    print(boxes)
    print(gt_classes)

    # I am ignoring gt_overlaps and seg_areas for now
    return {
        'image': self.root + '/' + filename,
        'img_id': idx, 
        'width': width,
        'height': height,
        'boxes': boxes,
        'gt_classes': gt_classes,
        'gt_overlaps': overlaps,
        'flipped': False,
        'seg_areas': seg_areas}

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb
    """
    gt_roidb = [self._load_roi(data, idx)
                for idx, data in enumerate(self._image_data)]

    """
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    """
    return gt_roidb


  def _get_widths(self):
    filenames = map(lambda d : d["filename"], self._image_data)

    widths = list()
    for f in filenames:
      im = Image.open(self.root + '/' + f)
      w, _ = im.size
      widths += [w]

    return widths

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

      self.roidb.append(entry)
    self._image_index = self._image_index * 2


  def evaluate_detections(self, all_boxes, output_dir):
    afasfas
    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     self._year +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_coco_results_file(all_boxes, res_file)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      self._do_detection_eval(res_file, output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)
