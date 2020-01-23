import os
import cv2
import argparse
import pandas as pd
import numpy as np
from glob import glob

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default='/cluster/scratch/aabhinav/rsna_data', help="Directory path where raw train label csv file is saved")

ARGS = parser.parse_args()

DATA_DIR = ARGS.data_dir


def id_split(x, n):
    return x.split('_')[n]

labels_df = pd.read_csv(os.path.join(DATA_DIR, 'stage_2_train.csv'))
labels_df.drop_duplicates(inplace=True)

labels_df['Sub_type'] = labels_df['ID'].apply(id_split, n=-1)
labels_df['PatientID'] = labels_df['ID'].apply(id_split, n=1)
labels_df['ImgID'] = labels_df['PatientID'].apply(lambda x: 'ID_'+x)

reshaped_df = labels_df[['ImgID', 'Sub_type', 'Label']]
reshaped_df = reshaped_df.pivot(index='ImgID', columns='Sub_type', values='Label')


labels_df.to_csv(os.path.join(DATA_DIR, 'processed_stage_2_train.csv'), index=False)
reshaped_df.to_csv(os.path.join(DATA_DIR, 'reshaped_stage_2_train.csv'), index=False)
