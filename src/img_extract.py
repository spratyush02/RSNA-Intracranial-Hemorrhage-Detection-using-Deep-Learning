import pydicom
import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
from zipfile import ZipFile

parser = argparse.ArgumentParser()

parser.add_argument("--zip_dir", type=str, default='/cluster/scratch/aabhinav/rsna_data', help="Directory path where raw zip files is saved")

ARGS = parser.parse_args()

DATA_DIR = ARGS.zip_dir


def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    '''
    Rescale image from 16 bit HU values to 8bit pixel values from 0 to 255. Put rescale=True if want pixel values between 0-255
    '''

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        img = (img - img_min) / (img_max - img_min)*255
    
    return img


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    '''
    Get key windowing parameters like window center, window width, intercept and slope
    '''
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def extract_image(dcm, fpath):
    '''
    Extract image from dcm file and save it in fpath
    '''
    window_center , window_width, intercept, slope = get_windowing(dcm)
    img = dcm.pixel_array

    is_train = 'train' in fpath

    img_init = window_image(img, window_center, window_width, intercept, slope, rescale=True)
    brain_img = window_image(img, 40, 80, intercept, slope, rescale=True)
    subdural_img = window_image(img, 80, 200, intercept, slope, rescale=True)
    bone_img = window_image(img, 600, 2800, intercept, slope, rescale=True)

    combined_img = np.stack([brain_img,subdural_img,bone_img], axis=-1)
    resized = cv2.resize(combined_img, (224, 224))
        
    if img_pct_window(img_init, window_center, window_width)>0.02 and is_train:
        cv2.imwrite(fpath, resized)
    if not is_train:
        cv2.imwrite(fpath, resized)


def fix_pixrepr(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

    return dcm

def img_pct_window(img, window_center, window_width):
    correct_pxl = np.sum((img > (window_center-window_width//2)) & (img < (window_center+window_width//2)))
    total_pxl = img.shape[0]*img.shape[1]
    return correct_pxl/total_pxl


corrupt = []
with ZipFile(os.path.join(DATA_DIR, 'rsna-intracranial-hemorrhage-detection.zip'), 'r') as f:
    namelist = f.namelist()
    for name in tqdm(namelist):
        try:
            if name.endswith('.dcm'):
                f.extract(name)
                dcm = pydicom.read_file(name)
                os.remove(name)

                folder_path = os.path.join(DATA_DIR, '/'.join(name.split('/')[:-1]))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                fpath = os.path.join(DATA_DIR, name[:-4]+'.jpg')

                if dcm.PixelRepresentation == 0 and dcm.RescaleIntercept>-100:
                    dcm = fix_pixrepr(dcm)

                extract_image(dcm, fpath)
        except ValueError:
            corrupt.append(name)

print(len(corrupt))