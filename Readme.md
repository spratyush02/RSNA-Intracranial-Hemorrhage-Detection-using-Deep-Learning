# RSNA Intracranial Hemorrhage Detection Instructions

## Table of contents
<!-- [Setup](#Setup)<br> -->
<!-- [File Structure](#Structure)<br> -->
<a href="#Setup">Setup</a><br>
<a href="#Structure">File Structure</a><br>
<a href="#Download">Files to be downloaded</a><br>
<a href='#Data'>Data preprocessing and preparation</a><br>
<a href='#Metadata'>Extracting metadata from DICOM files</a><br>
<a href='#Labels'>Processing Train Labels</a><br>
<a href='#Tfrecord'>Converting raw images to tfrecords format</a><br>
<a href='#BaseModel'>Training the Base model</a><br>
<a href='#Embeddings'>Extract base model embeddings for Sequence model</a><br>
<a href='#LSTM'>Training Sequence (LSTM) model</a><br>
<a href='#Prediction'>Prediction using saved checkpoint</a><br>

## Setup
<a name='Setup'></a>


**Required Python >= 3.6.0**

Create virtual environment

```
python -m venv .venv
```

Activate environment (Windows)

```
.venv\Scripts\activate.bat
```

Activate environment (Unix or MacOS)

```
source .venv/bin/activate
```

Installing dependencies (for CPU)

```
pip install -r requirements.cpu.txt
```

Installing dependencies (for GPU, recommended)

```
pip install -r requirements.gpu.txt
```

## File Structure
<a name='Structure'></a>

### **Save all the files in same folder except wherever mentioned. Also do not change the names of downloaded files**

Following should the file structure to run the model smoothly
```Shell
data/
    intracranial-hemorrhage-detection.zip (181G)
    rsna-intracranial-hemorrhage-detection/ (~13G)
        stage_2_train/
        stage_2_test/
    test/
        test.tfrecord (1.8G)
    train/
        train-01-of-20.tfrecord (545M)
        train-01-of-20.tfrecord
        .....
    validation/
        train-19-of-20.tfrecord (545M)
        train-20-of-20.tfrecord
    train_metadata.csv (80M)
    test_metadata.csv (13M)
    stage_2_train.csv (120M)
    processed_stage_2_train.csv (263M)
    reshaped_stage_2_train.csv (19M)
    train_embeddings.npy (14G)
    train_labels.npy (27M)
    train_mask.npy (9M)
    test_embeddings.npy (2.5G)
    test_img_ids.npy (9.7M)
experiments/
    1577737599_tf_bin_loss_b32_e10/
    lstm/
        1578060580_b32_e20_l3_cs128/
src/
    create_metadata.py
    .....
inception_resnet_v2_2016_08_30.ckpt (237M)
resnet_v2_50.ckpt (307M)
```

## Files to be downloaded
<a name='Download'></a>
- [Exracted Images](https://drive.google.com/file/d/1IhdTTZaf_yA-68pPKP49BIU9fk3R1Wy2/view?usp=sharing)
- [Metadata](https://drive.google.com/drive/folders/1TMa-F5tjuo85FIX86HEFt3BEO5lgy4-b?usp=sharing)
- [Train Labels](https://drive.google.com/file/d/1AaRD6fRbgfUOxy94ytP86b9pXcOosHa1/view?usp=sharing)
- [Processed Labels](https://drive.google.com/drive/folders/1J_zwOolMcG8u_v5U7lws5UICDWg1RL89?usp=sharing)
- [TFrecord Files](https://drive.google.com/drive/folders/1SPlqhdHLOcc3xVffuIai1U1GiS1yuldk?usp=sharing)
- [Pretrained Checkpoints](https://drive.google.com/drive/folders/1EnI5UMA20htug9K2SDZaYAp1-bEJysqs?usp=sharing)
- [Base Model Checkpoint](https://drive.google.com/open?id=1OBw9UMlbaJv7TkWhwS7Q-pYOMl0z8z-P)
- [Sequence Model Data](https://drive.google.com/open?id=13ZruHTRfwfZe6SN9YS4oiNsXpy6fXjd_)
- [Sequence Model Checkpoint](https://drive.google.com/drive/folders/11vP-KY9P1XqDC4V4GLWExwkGGLrNdp-X?usp=sharing)

## Data preprocessing and preparation
<a name='Data'></a>
**Requires 5G memory**<br>
To extract train and test images from DICOM files, download the raw `rsna-intracranial-hemorrhage-detection.zip` file [here](https://drive.google.com/drive/folders/1i_zr9yajYq7pz4TOAJQAddsJm9wbUnUt?usp=sharing). After this run the following command
```
python src/img_extract.py --zip_dir <zip file directory path>
```
- zip_dir: Directory path where zip file is present. **Please note that all the future data will be stored in this directory**

After this train images will be stored in `zip_dir/rsna-intracranial-hemorrhage-detection/stage_2_train` folder and test images will be stored in `zip_dir/rsna-intracranial-hemorrhage-detection/stage_2_test` folder. Since this process takes some so extracted images can be downloaded [here](https://drive.google.com/file/d/1IhdTTZaf_yA-68pPKP49BIU9fk3R1Wy2/view?usp=sharing) and save it in `zip_dir`

## Extracting metadata from DICOM files
<a name='Metadata'></a>
**Requires 5G memory**<br>
To extract train and test metadata, make sure that you have downloaded raw .zip file. Then run the following command
```
python src/create_metadata.py --zip_dir <zip file directory path>
```
- zip_dir: Directory path where zip file is present. **Please note that all the future data will be stored in this directory**

After this `train_metadata.csv` and `test_metadata.csv` will be created in zip_dir. These files can be downloaded [here](https://drive.google.com/drive/folders/1TMa-F5tjuo85FIX86HEFt3BEO5lgy4-b?usp=sharing) and save it in `zip_dir`

## Processing Train Labels
<a name='Labels'></a>
To process train labels, so that it can be fed to tfrecords creation step, download the train labels files [here](https://drive.google.com/file/d/1AaRD6fRbgfUOxy94ytP86b9pXcOosHa1/view?usp=sharing) and save it in same folder where raw .zip file is saved (`zip_dir`). The run the following command
```
python src/process_labels.py --data_dir <train label csv file directory>
```
- data_dir: Directory where train label csv file is saved

After this `processed_stage_2_train.csv` and `reshaped_stage_2_train.csv` will be created in data_dir. These files can be downloaded [here](https://drive.google.com/drive/folders/1J_zwOolMcG8u_v5U7lws5UICDWg1RL89?usp=sharing) and save it in `data_dir`

## Converting raw images to tfrecords format
<a name='Tfrecord'></a>
To convert raw images and labels to tfrecords files, make sure you have extracted images from dicom files and have also processed the train labels csv file (`processed_stage_2_train.csv`) as explained in previous step in same data directory. After all the raw data is prepared, run the following command
```
python src/create_tf_records.py --data_dir <directory path where all the data is being stored>
```
- data_dir: Directory path where all the data including tfrecords files will be saved

After this three folder naming `train`, `validation` and `test` will be created in `data_dir` and you can find tfrecord files there. Since this process takes some time, so you can download already created tfrecord files [here](https://drive.google.com/drive/folders/1SPlqhdHLOcc3xVffuIai1U1GiS1yuldk?usp=sharing) and save it in `data_dir`

## Training the Base model
<a name='BaseModel'></a>
**Requires 10G memory**<br>
To train the model you need to have saved checkpoint for **resnet_v2** and **inception_resnet**. Both checkpoints which can be downloaded from [here](https://drive.google.com/drive/folders/1EnI5UMA20htug9K2SDZaYAp1-bEJysqs?usp=sharing). **Download both the checkpoints in current working directory**. Then run following command
```
python src/train.py [--batch_size] [--num_epochs] [--lr] [--pretrained_model] [--save_dir] [--data_dir] [--loss] [--export]
```
- batch_size: default values is 32
- num_epochs: default values is 5
- lr: default value is 0.0001. Learning rate for training
- pretrained_model: default value is inception_resnet. Other possible value is resnet_50
- save_dir: default value is "./experiments". Directory where model and config files are saved
- data_dir: Directory path where all the data including tfrecords files are saved
- loss: default value is "bin_loss". Loss to be used for training
- export: Save model checkpoint or not?

After running this model checkpoints, config file and tensorboard files will be stored in directory naming save_dir/\<timestamp>\_tf_\<loss_name>\_b<batch_size>_e<epoch_num>

To launch the tensorboard file while training run the following command
```
tensorboard --logdir current/working/directory/save_dir/<timestamp>_tf_<loss_name>_b<batch_size>_e<epoch_num>
```

Saved checkpoints for base model training can be downloaded [here](https://drive.google.com/open?id=1OBw9UMlbaJv7TkWhwS7Q-pYOMl0z8z-P)

## Extract base model embeddings for Sequence model
<a name='Embeddings'></a>
**Requires 70G memory**<br>
To extract base model embeddings which can then be used for sequence model, make sure you have base model checkpoints saved in `current/working/directory/experiments/<timestamp>_*`. Also tfrecord files, metadata csv files and reshaped labels csv file (`reshaped_stage_2_train.csv`) is present in `data_dir`. Then run the following command
```
python src/extract_features.py --model_id <saved model id> [--base_data] [--max_to_keep] --data_dir <saved data directory path>
```
- model_id: model_id (timestamp) of saved model
- base_data: Whether to extract train data embeddings or test data embeddings. Default value is train. Possible values are train and test
- max_to_keep: How many number of timestamps data is needed in unrolled sequence model. Default is 60
- data_dir: Directory where tfrecord files, metadata csv files and reshaped labels csv file (`reshaped_stage_2_train.csv`) is saved

After this different files will be stored in data directory based on whether base_data is train or test.
- If base_data is train, then three `.npy` files will be created naming:
    - `train_embeddings.npy`: Embedding tensor of shape (patient_count, max_to_keep, base_model_final_layer_size). base_model_final_layer_size = 1536 for inception resnet and 2048 for resnet50 model.
    - `train_labels.npy`: Label tensor of shape (patient_count, max_to_keep, 6)
    - `train_mask.npy`: Mask tensor of shape (patient_count, max_to_keep). It has binary values having zero where padding is used else one.
If base_data is test, then two `.npy` files will be created naming:
    - `test_embeddings.npy`: Same as `train_embedding.npy`
    - `test_img_ids.npy`: Tensor of image ids of shape (patient_count, max_to_keep). It will later be used to map final predictions with image ids

Since this step can take some time, so all the required npy files can be downloaded [here](https://drive.google.com/open?id=13ZruHTRfwfZe6SN9YS4oiNsXpy6fXjd_) and save it in `data_dir`

## Training Sequence (LSTM) model
<a name='LSTM'></a>
**Requires 70G memory**<br>
To train Sequence (LSTM) model make sure `train_embeddings.npy`, `train_labels.npy` and `train_mask.npy` is saved in data_dir. Then run the following command:
```
python src/train_lstm.npy [--batch_size] [--num_epochs] [--num_layers] [--cell_size] [val_ratio] [--lr] [--dropout_rate] [--save_dir] [--data_dir] [--loss] [--export]]
```
- batch_size: default values is 32
- num_epochs: default values is 10
- num_layers: default value is 2
- cell_size: default value is 256
- val_ratio: default value is 0.1
- lr: default value is 0.0001. Learning rate for training
- dropout_rate: default value is 0.7
- save_dir: default value is "./experiments/lstm". Directory where model and config files are saved
- data_dir: Directory path where all the data including npy are saved
- loss: default value is "masked_bin_loss". Loss to be used for training
- export: Save model checkpoint or not?

After running this model checkpoints, config file and tensorboard files will be stored in directory naming save_dir/\<timestamp>\_b<batch_size>_e<epoch_num>\_l<num_layers>_cs<cell_size>

To launch the tensorboard file while training run the following command
```
tensorboard --logdir current/working/directory/save_dir/<timestamp>_b<batch_size>_e<epoch_num>_l<num_layers>_cs<cell_size>
```

Saved checkpoints for seqence model training can be downloaded [here](https://drive.google.com/drive/folders/11vP-KY9P1XqDC4V4GLWExwkGGLrNdp-X?usp=sharing). Save the checkpoints in `experiments/lstm/<timestamp>_*`


## Prediction using saved checkpoint
<a name='Prediction'></a>
**Requires 10G memory**<br>
To predict on test set make sure that you have saved checkpoint files, `test.tfrecord`, `test_embeddings.npy` and `test_img_ids.npy` in correct folders. Prediction can be made on base model as well as sequence model. Run the following command
```
python src/predict.py --model_id <saved model id> [--model_type] --data_dir <saved data directory path>
```
- model_id: model_id (timestamp) of saved model
- model_type: Whether to make prediction on base model or lstm model. Default value is lstm. Possible values are lstm and base
- data_dir: Directory path where all the data including npy are saved

After running this prediction file on test data will be generated in current working directory named `output2.csv`


Write about memory requirements and time overall bsub command. Also remove option of save dir