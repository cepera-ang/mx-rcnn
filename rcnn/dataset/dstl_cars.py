"""
DSTL CARS database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

from builtins import super
from builtins import dict
from builtins import open
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import range
import pickle
# import cv2
import os
import numpy as np
import pandas as pd

from .imdb import IMDB
from .ds_utils import unique_boxes, filter_small_boxes
from tqdm import tqdm



class dstl_cars(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: dstl_cars
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        super(dstl_cars, self).__init__('dstl_cars', image_set, root_path, devkit_path)  # set self.name

        self.simple_labels = pd.read_csv('D:/dstl_cars/{}.csv'.format(image_set))

        self.root_path = root_path
        self.devkit_path = devkit_path

        self.data_path = os.path.join('d:/patches_{}'.format(image_set))

        self.classes = ['__background__',  # always index 0
                        'A', 'B', 'C', 'D',
                        'E', 'F', 'G', 'H', 'I']

        self.num_classes = len(self.classes)
        self.image_set_index = self.simple_labels['file_name'].str.replace('.jpg', '').unique()
        self.num_images = len(self.image_set_index)
        print('num_images', self.num_images)

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}


    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        # image_file = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        image_file = os.path.join(self.data_path, index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self.load_pascal_annotation(index) for index in tqdm(self.image_set_index)]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def load_pascal_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from pandas simple_labels
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        roi_rec = dict()

        file_name = self.simple_labels.loc[self.simple_labels['file_name'].str.contains(index), 'file_name'].values[0]
        roi_rec['image'] = os.path.join(self.data_path, file_name)

        if self.image_set == 'dstl_test_1000':
            roi_rec['height'] = 1000
            roi_rec['width'] = 1000
        elif self.image_set == 'dstl_test_2000':
            roi_rec['height'] = 2000
            roi_rec['width'] = 2000

        else:
            roi_rec['height'] = 700
            roi_rec['width'] = 700

        df_c = self.simple_labels[self.simple_labels['file_name'] == file_name].reset_index()

        num_objs = df_c.shape[0]

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(list(zip(self.classes, list(range(self.num_classes)))))
        # Load object bounding boxes into a data frame.

        for ix in df_c.index:
            if self.image_set.find('test') != -1 or self.image_set == 'dstl_train_2000':
                x1 = 0
                y1 = 0
                x2 = 0
                y2 = 0
                class_name = '__background__'
            else:
                # Make pixel indexes 0-based
                x1 = float(df_c.loc[ix, 'x_min'])
                y1 = float(df_c.loc[ix, 'y_min'])
                x2 = float(df_c.loc[ix, 'x_max'])
                y2 = float(df_c.loc[ix, 'y_max'])
                class_name = df_c.loc[ix, 'class_name']

            cls = class_to_index[class_name]

            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    def load_selective_search_roidb(self, gt_roidb):
        """
        turn selective search proposals into selective search roidb
        :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        import scipy.io
        matfile = os.path.join(self.root_path, 'selective_search_data', self.name + '.mat')
        assert os.path.exists(matfile), 'selective search data does not exist: {}'.format(matfile)
        raw_data = scipy.io.loadmat(matfile)['boxes'].ravel()  # original was dict ['images', 'boxes']

        box_list = []
        for i in tqdm(list(range(raw_data.shape[0]))):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1  # pascal voc dataset starts from 1.
            keep = unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_roidb(self, gt_roidb, append_gt=False):
        """
        get selective search roidb and ground truth roidb
        :param gt_roidb: ground truth roidb
        :param append_gt: append ground truth
        :return: roidb of selective search
        """
        cache_file = os.path.join(self.cache_path, self.name + '_ss_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if append_gt:
            print('appending ground truth annotations')
            ss_roidb = self.load_selective_search_roidb(gt_roidb)
            roidb = IMDB.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self.load_selective_search_roidb(gt_roidb)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = os.path.join(self.devkit_path, 'results')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.devkit_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.devkit_path, 'results', 'dstl')
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.devkit_path, 'results', 'dstl', 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        self.do_python_eval()

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} DSTL results file'.format(cls))
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: None
        """
        # annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        # imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        # annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')
        # aps = []
        # # The PASCAL VOC metric changed in 2010
        # use_07_metric = True if int(self.year) < 2010 else False
        # print('VOC07 metric? ' + ('Y' if use_07_metric else 'No'))
        # for cls_ind, cls in enumerate(self.classes):
        #     if cls == '__background__':
        #         continue
        #     filename = self.get_result_file_template().format(cls)
        #     rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
        #                              ovthresh=0.5, use_07_metric=use_07_metric)
        #     aps += [ap]
        #     print('AP for {} = {:.4f}'.format(cls, ap))
        # print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print("Python Eval")
