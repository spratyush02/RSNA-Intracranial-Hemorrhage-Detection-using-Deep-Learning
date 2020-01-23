import os
import pydicom
import argparse
import pandas as pd
from tqdm import tqdm
from zipfile import ZipFile

parser = argparse.ArgumentParser()

parser.add_argument("--zip_dir", type=str, default='/cluster/scratch/aabhinav/rsna_data', help="Directory path where raw zip files is saved")


ARGS = parser.parse_args()

DATA_DIR = ARGS.zip_dir

def generate_dict(dcm, data_dict):
    '''
    Returns meta data dictionary with keys as feature name and values as array of corresponding values
    '''
    
    all_keywords = dcm.dir()
    ignored = ['Rows', 'Columns', 'PixelData', 'ImageOrientationPatient', 'PixelSpacing', 'WindowCenter', 'WindowWidth']

    for name in all_keywords:
        if name in ignored:
            continue

        if name not in data_dict:
            data_dict[name] = []

        if name == 'ImagePositionPatient':
            data_dict[name].append(dcm[name].value[-1])
        else:
            data_dict[name].append(dcm[name].value)
    
    return data_dict


with ZipFile(os.path.join(DATA_DIR, 'rsna-intracranial-hemorrhage-detection.zip'), 'r') as f:
    namelist = f.namelist()
    train_dict = {}
    test_dict = {}

    for name in tqdm(namelist):
        try:
            if name.endswith('.dcm'):
                f.extract(name)
                dcm = pydicom.read_file(name)
                os.remove(name)

                if 'train' in name:
                    train_dict = generate_dict(dcm,train_dict)
                if 'test' in name:
                    test_dict = generate_dict(dcm,test_dict)

                # break
        except ValueError:
            pass

    train_df = pd.DataFrame(train_dict)
    test_df = pd.DataFrame(test_dict)

    train_df.to_csv(os.path.join(DATA_DIR, 'train_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR,'test_metadata.csv'), index=False)
