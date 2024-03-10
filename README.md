# Intelligence-Hub-
six.moves.urllib
urllib.request
RoadDamageDataset.tar.gz
https://s3-ap-northeast-1.amazonaws.com/mycityreport/trainedModels.tar.gz
import six.moves.urllib as urllib try:
    import urllib.request except ImportError:     raise ImportError('You should use Python 3.x')
if not os.path.exists('./RoadDamageDataset.tar.gz'):
    url_base = 'https://s3-ap-northeast-1.amazonaws.com/mycityreport/RoadDamageDataset.tar.gz'     urllib.request.urlretrieve(url_base, './RoadDamageDataset.tar.gz')
    print("Download RoadDamageDataset.tar.gz Done")
else:
    print("You have RoadDamageDataset.tar.gz") if not os.path.exists('./trainedModels.tar.gz'):
    url_base = 'https://s3-ap-northeast-1.amazonaws.com/mycityreport/trainedModels.tar.gz'     urllib.request.urlretrieve(url_base, './trainedModels.tar.gz')
    print("Download trainedModels.tar.gz Done")
else:
    print("You have trainedModels.tar.gz")
!tar -zxf ./RoadDamageDataset.tar.gz !tar -zxf ./trainedModels.tar.gz from xml.etree import ElementTree from xml.dom import minidom import collections
import os
import matplotlib.pyplot as plt import matplotlib as matplot import seaborn as sns %matplotlib inline base_path = os.getcwd() + '/RoadDamageDataset/'
damageTypes=["D00", "D01", "D10", "D11", "D20", "D40", "D43", "D44"]
# govs corresponds to municipality name. govs = ["Adachi", "Chiba", "Ichihara", "Muroran", "Nagakute", "Numazu", "Sumida"] cls_names = [] total_images = 0 for gov in govs:
    file_list = [filename for filename in os.listdir(base_path + gov + '/Annotations/') if not filename.startswith('.')]
    for file in file_list:
        total_images = total_images + 1         if file =='.DS_Store':
            pass         else:
            infile_xml = open(base_path + gov + '/Annotations/' +file)             tree = ElementTree.parse(infile_xml)             root = tree.getroot()             for obj in root.iter('object'):
                cls_name = obj.find('name').text                 cls_names.append(cls_name) print("total") print("# of images：" + str(total_images)) print("# of labels：" + str(len(cls_names))) import collections count_dict = collections.Counter(cls_names) cls_count = [] for damageType in damageTypes:
    print(str(damageType) + ' : ' + str(count_dict[damageType]))     cls_count.append(count_dict[damageType])
sns.set_palette("winter", 8) sns.barplot(damageTypes, cls_count) for gov in govs:     cls_names = []     total_images = 0
    file_list = [filename for filename in os.listdir(base_path + gov + '/Annotations/') if not filename.startswith('.')]
    for file in file_list:
        total_images = total_images + 1         if file =='.DS_Store':
            pass         else:
            infile_xml = open(base_path + gov + '/Annotations/' +file)             tree = ElementTree.parse(infile_xml)             root = tree.getroot()             for obj in root.iter('object'):                 cls_name = obj.find('name').text                 cls_names.append(cls_name)     print(gov)     print("# of images：" + str(total_images))     print("# of labels：" + str(len(cls_names)))
    count_dict = collections.Counter(cls_names)     cls_count = []     for damageType in damageTypes:
        print(str(damageType) + ' : ' + str(count_dict[damageType]))         cls_count.append(count_dict[damageType])
    print('**************************************************') import cv2 import random
def draw_images(image_file):     gov = image_file.split('_')[0]     img = cv2.imread(base_path + gov + '/JPEGImages/' + image_file.split('.')[0] + '.jpg')
    infile_xml = open(base_path + gov + '/Annotations/' +image_file)     tree = ElementTree.parse(infile_xml)     root = tree.getroot()
    for obj in root.iter('object'):         cls_name = obj.find('name').text         xmlbox = obj.find('bndbox')         xmin = int(xmlbox.find('xmin').text)         xmax = int(xmlbox.find('xmax').text)         ymin = int(xmlbox.find('ymin').text)         ymax = int(xmlbox.find('ymax').text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # put text
        cv2.putText(img,cls_name,(xmin,ymin-10),font,1,(0,255,0),2,cv2.LINE_AA)
        # draw bounding box         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0),3)     return img for damageType in damageTypes:
    tmp = []     for gov in govs:
        file = open(base_path + gov + '/ImageSets/Main/%s_trainval.txt' %damageType, 'r')
        for line in file:             line = line.rstrip('\n').split('/')[-1]
            if line.split(' ')[2] == '1':                 tmp.append(line.split(' ')[0]+'.xml')
    random.shuffle(tmp)     fig = plt.figure(figsize=(6,6))     for number, image in enumerate(tmp[0:1]):
        img = draw_images(image)         plt.subplot(1,1,number)
        plt.axis('off')
        plt.title('The image including ' + damageType)         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) import numpy as np import sys import tarfile import tensorflow as tf import zipfile
from collections import defaultdict from io import StringIO from matplotlib import pyplot as plt from PIL import Image
if tf.__version__ != '1.4.1':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.1!') from utils import label_map_util
from utils import visualization_utils as vis_util
PATH_TO_CKPT =  'trainedModels/ssd_mobilenet_RoadDamageDetector.pb'
PATH_TO_LABELS = 'trainedModels/crack_label_map.pbtxt'
NUM_CLASSES = 8
detection_graph = tf.Graph() with detection_graph.as_default():   od_graph_def = tf.GraphDef()   with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()     od_graph_def.ParseFromString(serialized_graph)     tf.import_graph_def(od_graph_def, name='') label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True) category_index = label_map_util.create_category_index(categories) def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size   return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/DATASET/CrackDataset-TF/CrackDataset/' D_TYPE = ['D00', 'D01', 'D10', 'D11', 'D20','D40', 'D43'] govs = ['Adachi', 'Ichihara', 'Muroran', 'Chiba', 'Sumida', 'Nagakute', 'Numazu']
val_list = [] for gov in govs:
    file = open(PATH_TO_TEST_IMAGES_DIR + gov + '/ImageSets/Main/val.txt', 'r')
    for line in file:
        line = line.rstrip('\n').split('/')[-1]         val_list.append(line)     file.close()
print("# of validation images：" + str(len(val_list))) TEST_IMAGE_PATHS=[]
random.shuffle(val_list)
for val_image in val_list[0:5]:
    TEST_IMAGE_PATHS.append(PATH_TO_TEST_IMAGES_DIR + val_image.split('_')[0]+ 
'/JPEGImages/%s.jpg' %val_image) IMAGE_SIZE = (12, 8) with detection_graph.as_default():   with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')     detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')     detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')     detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')     num_detections = detection_graph.get_tensor_by_name('num_detections:0')     for image_path in TEST_IMAGE_PATHS:       image = Image.open(image_path)       image_np = load_image_into_numpy_array(image)       image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],           feed_dict={image_tensor: image_np_expanded})       vis_util.visualize_boxes_and_labels_on_image_array(           image_np,           np.squeeze(boxes),           np.squeeze(classes).astype(np.int32),           np.squeeze(scores),           category_index,           min_score_thresh=0.3,           use_normalized_coordinates=True,           line_thickness=8)       plt.figure(figsize=IMAGE_SIZE)       plt.imshow(image_np)
      !rm -rf ./RoadDamageDataset.tar.gz
!rm -rf ./RoadDamageDataset
!rm -rf ./trainedModels.tar.gz
!rm -rf ./trainedModels
