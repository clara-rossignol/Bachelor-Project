import pandas as pd
import numpy as np
import random
import deeplabcut
import math
from IPython.display import Image, display


# corrupting data from a csv deeplabcut file
def corruption_data(original,scorer,bodyparts,list_i,n):
    df = pd.read_csv(original,header=[0,1,2],index_col=0)
    filenames = df.index.tolist()
    random.shuffle(filenames)
    corrupted_frames = filenames[:n]

    for name in corrupted_frames:
        for i in list_i:
            df.loc[name][(scorer,bodyparts[i], 'x')] = np.random.uniform(5,200)
            df.loc[name][(scorer,bodyparts[i], 'y')] = np.random.uniform(5,100)
            
    return df,corrupted_frames


#swap two variables
def swap(x,y):
    tmp = x
    x = y
    y = tmp
    return x,y


# swap labels from biologically symmetric body parts
def swap_labels(original,scorer,bodyparts,i,j,n):
    df = pd.read_csv(original,header=[0,1,2],index_col=0)
    filenames = df.index.tolist()
    random.shuffle(filenames)
    swapped_frames = filenames[:n]
    
    for name in swapped_frames:
        df.loc[name][(scorer,bodyparts[i],'x')],df.loc[name][(scorer,bodyparts[j],'x')] = swap(df.loc[name][(scorer,bodyparts[i],'x')],df.loc[name][(scorer,bodyparts[j],'x')])
        df.loc[name][(scorer,bodyparts[i],'y')],df.loc[name][(scorer,bodyparts[j],'y')] = swap(df.loc[name][(scorer,bodyparts[i],'y')],df.loc[name][(scorer,bodyparts[j],'y')])
    
    return df,swapped_frames


# calculate the distance between 2 keypoints
def distance_keypoints(df,kpt1,kpt2):
    bdpt1 = df.xs(kpt1, level='bodyparts', axis=1).to_numpy()
    bdpt2 = df.xs(kpt2, level='bodyparts', axis=1).to_numpy()
    diff = (bdpt1-bdpt2).reshape(len(df), -1, 2)
    dist = np.linalg.norm(diff, axis=2)
    return dist


# calculate the angle between 3 keypoints (if negative, swapping might have occured)
def getAngle(a,b,c):
    angle = math.degrees(math.atan2(c[1]-b[1],c[0]-b[0]) - math.atan2(a[1]-b[1],a[0]-b[0]))
    return angle


# remove any NaN values from our dataset
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)