#%%
import pydicom
import cv2

#%%
path = 'data/ID_000039fa0.dcm'
fname = 'ct_image1.jpg'

dcm = pydicom.read_file(path)
# cv2.imwrite(fname, ds.pixel_array)


#%%
def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def fix_pixrepr(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

    return dcm

# TODO: Add pixel spacing normalization

#%%
window_center , window_width, intercept, slope = get_windowing(dcm)
img = dcm.pixel_array
img = window_image(img, window_center, window_width, intercept, slope, rescale=True)

resized = cv2.resize(img, (224, 224))
res = cv2.imwrite(fname, resized)

#%%
img = cv2.imread('ct_image1.jpg', 0)

#%%
