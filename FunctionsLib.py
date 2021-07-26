import pandas as pd
import numpy as np
import random
import deeplabcut
from IPython.display import Image, display


# corrupting data from a csv deeplabcut file
def corruption_data(original,scorer,bodyparts,list_i,n):
    df = pd.read_csv(original,header=[0,1,2],index_col=0)
    filenames = df.index.tolist()
    random.shuffle(filenames)
    corrupted_frames = filenames[:n]

    for name in corrupted_frames:
        for i in list_i:
            df.loc[name][(scorer,bodyparts[i], 'x')] = np.random.uniform(5,500)
            df.loc[name][(scorer,bodyparts[i], 'y')] = np.random.uniform(5,500)
            
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


# calculate the angle between 3 points
def getAngle(a,b,c):
    angle = math.degrees(math.atan2(c[1]-b[1],c[0]-b[0]) - math.atan2(a[1]-b[1],a[0]-b[0]))
    return angle
