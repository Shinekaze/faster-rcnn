import os
import cv2
import glob
from cv2 import COLOR_BGR2GRAY
import numpy as np
import fnmatch
from scipy.cluster.vq import kmeans, vq
from sklearn.model_selection import train_test_split
import pickle


def load_mvtec_dataset(directory, object_type, resize_dim=0):
    """
    Loads the MVTEC dataset of the specified object type. (bottle, cable, capsule, etc.)

    :param str directory: The relative path to the dataset directory.
    :param str object_type: The object type to load. (bottle, cable, capsule, etc.)
    :param int resize_dim: The dimension to resize the images to. Default image sizes are 900x900 (0 for no resizing).
    :return: A tuple containing the training images, training labels, testing images, and testing labels.
    """

    train_paths, test_paths, ground_truth_paths = _filter_mvtec_dataset_paths(directory, object_type)
    train_images, train_labels, test_images, test_labels = _load_and_label_data((train_paths, test_paths), resize_dim)
    return train_images, train_labels, test_images, test_labels, ground_truth_paths


def _filter_mvtec_dataset_paths(directory, object_type):
    """
    Internal function. Shoudln't be called directly.
    
    Recursively walks through the entire dataset directory and returns the paths to all images of the specified object type.

    :param str directory: The relative path to the dataset directory.
    :param str object_type: The object type to load. (bottle, cable, capsule, etc.)
    """

    # Check if directory exists
    if not os.path.isdir(directory):
        raise Exception(f"Can't load dataset: Directory \"{os.path.abspath(directory)}\" does not exist.")

    ground_truth_paths = {}
    test_paths = {'good': [], 'bad': []}
    train_paths = {'good': [], 'bad': []}

    # Extract all directories that only contain images with the specified object type
    filtered_paths = []
    for abs_path, sub_dirs, files in os.walk(directory):
        if len(sub_dirs) == 0:
            split_abs_path = abs_path.split('\\')
            if split_abs_path[1] == object_type:
                filtered_paths.append(abs_path)
    
    # Extract all image paths from the filtered directories
    for filtered_path in filtered_paths:
        split_abs_path = filtered_path.split('\\')
        type = split_abs_path[2]
        spec = split_abs_path[3]

        path_helper_func = lambda x: os.path.join(filtered_path, x)

        if type == 'ground_truth':
            ground_truth_paths[spec] = list(map(path_helper_func, os.listdir(filtered_path)))
        elif type == 'test':
            test_paths['bad' if spec != 'good' else 'good'].extend(list(map(path_helper_func, os.listdir(filtered_path))))
        elif type == 'train':
            train_paths['bad' if spec != 'good' else 'good'].extend(list(map(path_helper_func, os.listdir(filtered_path))))

    return train_paths, test_paths, ground_truth_paths


def _load_and_label_data(dataset, resize_dim=0):
    """
    Internal function. Shoudln't be called directly.

    Loads and labels the specified dataset.

    :param tuple dataset: A tuple containing the training and testing image paths.
    :param int resize_dim: The dimension to resize the images to. Default image sizes are 900x900 (0 for no resizing).
    """
    train_paths, test_paths = dataset

    # Load all training images
    train_images = []
    train_labels = []
    for path in train_paths['good']:
        image = cv2.imread(path)
        #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        train_images.append(image)
        train_labels.append(1)
    
    
    for path in train_paths['bad']:
        image = cv2.imread(path)
        #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        train_images.append(image)
        train_labels.append(0)
    
    # Load all testing images
    test_images = []
    test_labels = []
    for path in test_paths['good']:
        image = cv2.imread(path)
        #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        test_images.append(image)
        test_labels.append(1)
    
    for path in test_paths['bad']:
        image = cv2.imread(path)
        #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        test_images.append(image)
        test_labels.append(0)
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

def get_image_features(images, k=200):
    """
    Returns the image features for the specified images.
    :param images: The images to extract features from.
    :param k: The number of clusters to use.
    :return image_features: The extracted image features.
    """

    orb = cv2.ORB_create()
    desc_list = []
    kp_list = []

    # Get image keypoints and descriptors
    for img in images:
        kp = orb.detect(img, None)
        kp, descriptor = orb.compute(img, kp)
        desc_list.append(descriptor)
        kp_list.append(kp)

    kp_list = np.array(kp_list) 
    desc_list = np.array(desc_list)

    # Reshape descriptors to be a list of vectors
    descriptors = desc_list[0][1]
    for descriptor in desc_list[1:]:
        descriptors=np.vstack((descriptors,descriptor))
    
    # Apply k-means clustering to the descriptors
    voc, variance = kmeans(descriptors.astype(float), k, 1)

    # Compute the histogram of features
    img_features = np.zeros((len(images), k), "float32")
    for i in range(len(images)):
        words, distance = vq(desc_list[i], voc)
        for w in words:
            img_features[i][w] += 1

    return img_features


# load dataset using image data generator
def load_dataset(directory, object_type, resize_dim=0):
    categories = ['train', 'test', 'ground_truth']
    main_path = os.path.join(directory, object_type)
    
    dataset = {}
    for category in categories:
        path = os.path.join(main_path, category)
        dataset[category] = (tf.keras.utils.image_dataset_from_directory(path, image_size=(resize_dim, resize_dim), seed=123, batch_size=32))
    
    return dataset


def recopy_mvtec_resnet(dest_directory, img_directory, create_dir, MVTEC_CATEGORIES, test_val_size, resize_dim=0):

    if create_dir == 1: 
        os.mkdir(f"{dest_directory}")
    for cat in MVTEC_CATEGORIES:
        if create_dir == 1:
            os.mkdir(f"{dest_directory}/{cat}")
            os.mkdir(f"{dest_directory}/{cat}/train")
            os.mkdir(f"{dest_directory}/{cat}/train/good")
            os.mkdir(f"{dest_directory}/{cat}/train/bad")
            os.mkdir(f"{dest_directory}/{cat}/test")
            os.mkdir(f"{dest_directory}/{cat}/test/good")
            os.mkdir(f"{dest_directory}/{cat}/test/bad")
            os.mkdir(f"{dest_directory}/{cat}/val")
            os.mkdir(f"{dest_directory}/{cat}/val/good")
            os.mkdir(f"{dest_directory}/{cat}/val/bad")
        train_images, train_labels, test_images, test_labels, ground_truth_paths = load_mvtec_dataset(img_directory, cat)
        print(cat)
        good_images = []
        for img in train_images:
            if resize_dim != 0:
                img = cv2.resize(img, (resize_dim, resize_dim))
            good_images.append(img)
        
        bad_images = []
        for i in range(len(test_images)):
            if resize_dim != 0:
                img = cv2.resize(test_images[i], (resize_dim, resize_dim))
            if test_labels[i] == 1:
                good_images.append(img)
            else:
                bad_images.append(img)
    
        good_train, good_rem = train_test_split(good_images, test_size=test_val_size, random_state=1)
        good_test, good_val = train_test_split(good_rem, test_size=0.5, random_state=1)

        for i in range(len(good_train)):
            cv2.imwrite(f"{dest_directory}/{cat}/train/good/{i:03}.png", good_train[i])
        for i in range(len(good_test)):
            cv2.imwrite(f"{dest_directory}/{cat}/test/good/{i:03}.png", good_test[i])
        for i in range(len(good_val)):
            cv2.imwrite(f"{dest_directory}/{cat}/val/good/{i:03}.png", good_val[i])
        
        bad_train, bad_rem = train_test_split(bad_images, test_size=test_val_size, random_state=1)
        bad_test, bad_val = train_test_split(bad_rem, test_size=0.5, random_state=1)

        for i in range(len(bad_train)):
            cv2.imwrite(f"{dest_directory}/{cat}/train/bad/{i:03}.png", bad_train[i])
        for i in range(len(bad_test)):
            cv2.imwrite(f"{dest_directory}/{cat}/test/bad/{i:03}.png", bad_test[i])
        for i in range(len(bad_val)):
            cv2.imwrite(f"{dest_directory}/{cat}/val/bad/{i:03}.png", bad_val[i])
        
    return True


def load_mvtec(directory, object_type, resize_dim=0):
    train_images = []
    train_labels = []
    train_path=(f"{directory}/{object_type}/train")
    for path in glob.glob(f"{train_path}/good/*"):
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        train_images.append(image)
        train_labels.append(1)
    
    
    for path in glob.glob(f"{train_path}/bad/*"):
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        train_images.append(image)
        train_labels.append(0)
    
    # Load all testing images
    test_images = []
    test_labels = []
    test_path=(f"{directory}/{object_type}/test")
    for path in glob.glob(f"{test_path}/good/*"):
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        test_images.append(image)
        test_labels.append(1)
    
    
    for path in glob.glob(f"{test_path}/bad/*"):
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        test_images.append(image)
        test_labels.append(0)
    
    # Load all validation images
    val_images = []
    val_labels = []
    val_path=(f"{directory}/{object_type}/val")
    for path in glob.glob(f"{val_path}/good/*"):
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        val_images.append(image)
        val_labels.append(1)
    
    
    for path in glob.glob(f"{val_path}/bad/*"):
        image = cv2.imread(path)
        if resize_dim != 0:
            image = cv2.resize(image, (resize_dim, resize_dim))
        val_images.append(image)
        val_labels.append(0)
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels), np.array(val_images), np.array(val_labels)


def recopy_mvtec_yolo(dest_directory, img_directory, create_dir, MVTEC_CATEGORIES, test_val_size, resize_dim=0):

    if create_dir == 1: 
        os.mkdir(f"{dest_directory}")
    for cat in MVTEC_CATEGORIES:
        if create_dir == 1:
            os.mkdir(f"{dest_directory}/{cat}")
            os.mkdir(f"{dest_directory}/{cat}/images")
            os.mkdir(f"{dest_directory}/{cat}/images/train")
            os.mkdir(f"{dest_directory}/{cat}/images/test")
            os.mkdir(f"{dest_directory}/{cat}/images/val")
            os.mkdir(f"{dest_directory}/{cat}/labels")
            os.mkdir(f"{dest_directory}/{cat}/labels/train")
            os.mkdir(f"{dest_directory}/{cat}/labels/val")
            
        train_images, train_labels, test_images, test_labels, ground_truth_paths = load_mvtec_dataset(img_directory, cat)
        print(cat)
       
        bad_images = []
        for i in range(len(test_images)):
            if resize_dim != 0:
                img = cv2.resize(test_images[i], (resize_dim, resize_dim))
            if test_labels[i] == 0:
                bad_images.append(img)
        
        bad_train, bad_rem = train_test_split(bad_images, test_size=test_val_size, random_state=1)
        bad_test, bad_val = train_test_split(bad_rem, test_size=0.5, random_state=1)

        for i in range(len(bad_train)):
            cv2.imwrite(f"{dest_directory}/{cat}/images/train/{i:03}.png", bad_train[i])
        for i in range(len(bad_test)):
            cv2.imwrite(f"{dest_directory}/{cat}/images/test/{i:03}.png", bad_test[i])
        for i in range(len(bad_val)):
            cv2.imwrite(f"{dest_directory}/{cat}/images/val/{i:03}.png", bad_val[i])
        
    return True


def _bbox_from_contours(contours):
    xmin, ymin = contours[0][0][0]
    xmax, ymax = contours[0][0][0]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        xmin, xmax = min(x, xmin), max(x+w, xmax)
        ymin, ymax = min(y, ymin), max(y+h, ymax)
    
    return xmin, ymin, xmax - xmin, ymax - ymin


def _extract_bbox(img, as_percent=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = _bbox_from_contours(contours)
    if as_percent:
        x = round(x / img.shape[1], 3)
        y = round(y / img.shape[0], 3)
        w = round(w / img.shape[1], 3)
        h = round(h / img.shape[0], 3)
    return x, y, w, h


def _annotate_detection_class(cls_path, debug):
    image_names = [x for x in os.listdir(cls_path) if 'debug' not in x]
    bboxs = []
    
    if debug:
        dpath = f"{cls_path}/debug"
        if not os.path.isdir(dpath):
            os.mkdir(dpath)
    
    imgs_abs_path = []
    for img_name in image_names:
        img_abs_path = os.path.join(cls_path, img_name)
        
        img = cv2.imread(img_abs_path)
        if img is None:
            print(img_abs_path)
            assert False
        bbox = _extract_bbox(img)
        
        if debug:
            cv2.rectangle(img, (int(bbox[0] * img.shape[1]), int(bbox[1] * img.shape[0])), (int((bbox[0] + bbox[2]) * img.shape[1]), int((bbox[1] + bbox[3]) * img.shape[0])), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(cls_path, 'debug', img_name), img)
        
        bboxs.append(bbox)
        imgs_abs_path.append(img_abs_path)
    return bboxs, imgs_abs_path


def _annotate_detection_object(do_folder, debug):    
    gt_path = os.path.join(do_folder, 'ground_truth')
    classes = list(os.walk(gt_path))[0][1]
    print(f'Found {len(classes)} classes in {gt_path}')
    print(f'Classes: {classes}') 
    
    class_annotations = {}   
    for idx, cls in enumerate(classes):
        class_annotations[idx] = _annotate_detection_class(os.path.join(gt_path, cls), debug)
        
    return class_annotations


def annotate_dataset(dataset_path, debug=False):
    detection_objects = list(os.walk(dataset_path))[0][1]
    
    annotation_objects = {}
    
    print(f'Found {len(detection_objects)} detection objects.')
    # print name of all detection objects
    print(f'Detection objects: {detection_objects}')
    for idx, do in enumerate(detection_objects):
        print(f'Annotating: {do} ({idx+1}/{len(detection_objects)})')
        annotation_objects[idx] = _annotate_detection_object(os.path.join(dataset_path, do), debug)
        
    return annotation_objects


def create_annotation_files(annotations):  
    if not os.path.isdir('annotations'):
        os.mkdir('annotations')
    
    for k_obj, v_obj in annotations.items():
        with open(f"annotations/{k_obj}.txt", "w") as f:
            for k_cls, v_cls in v_obj.items():
                for bbox, abs_path in zip(v_cls[0], v_cls[1]):
                    abs_path = abs_path.replace('ground_truth', 'test')
                    abs_path = abs_path.replace('_mask', '')                   
                    f.write(f'{k_cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {abs_path}\n')


def load_annotation_file(object_type):
    dataset = []
    with open(f"annotations/{object_type}.txt", "r") as f:
        for line in f:
            line = line.strip()
            cls, x, y, w, h, abs_path = line.split(' ')
            x, y, w, h = float(x), float(y), float(w), float(h)
            dataset.append((cls, (x, y, w, h), abs_path))
    
    return dataset

def train_test_split_annotations(dataset, train_size=0.8, validation_size=0.5):
    len_ds = len(dataset)
    
    train_ratio = int(len_ds*train_size)
    test_ratio= int(len_ds*(1-(1-train_size)*validation_size))
    val_ratio = test_ratio
    
    return dataset[:train_ratio], dataset[train_ratio:test_ratio], dataset[val_ratio:]