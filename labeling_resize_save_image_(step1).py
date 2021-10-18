import cv2
import os
import random
import numpy as np
import pickle

def OnehotEncodeing(arr):
    unique=[]
    result=[]
    for i in arr:
        if i not in unique:
            unique.append(i)
    for j in arr:
        zero=[0 for i in range(len(unique))]
        zero[unique.index(j)]=1.0
        result.append(zero)
    return result

CNN_training_data=[]
X=[]
Y=[]
img_size=224
directory="data1/"
class_types=["with_mask","without_mask"]
try:
    for class_name in class_types:
        path=os.path.join(directory,class_name)
        for im in os.listdir(path):
                img=cv2.imread(os.path.join(path,im))
                img_resize = cv2.resize(img,(img_size, img_size))
                CNN_training_data.append([img_resize,class_name])

    random.shuffle(CNN_training_data)
    for features,label in CNN_training_data:
        X.append(features)
        Y.append(label)

    X=np.array(X).reshape(-1,img_size,img_size,3)
    X=X/255.0
    Y=OnehotEncodeing(Y)
    Y=np.array(Y)
    

    write_X=open("X.pickle","wb")
    pickle.dump(X,write_X)
    write_X.close()
    write_Y=open("Y.pickle", "wb")
    pickle.dump(Y,write_Y)
    write_Y.close()
except Exception as e:
    pass
