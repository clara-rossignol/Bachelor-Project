import pandas as pd
import numpy as np
import random
import deeplabcut

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
