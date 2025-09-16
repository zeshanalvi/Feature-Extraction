import cv2, numpy as np, os
import pandas as pd
from skimage import feature
import warnings
warnings.filterwarnings('ignore')


from .image_read import Dataset


def get_lbp_single(image=None,radius=1,eps=1e-7):
  numPoints=4*radius
  #print(type(image))
  #image.astype(np.uint8)
  #print(image.shape,type(image[0,0,0]))
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  lbp = feature.local_binary_pattern(gray_image, numPoints,radius, method="uniform")
  (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
  hist = hist.astype("float")
  hist /= (hist.sum() + eps)
  return lbp, hist

def get_lbps(dataset,paths=None,label=None,storage_path=None,radii=[1,2,3,4,5],batch_size=1000):
  #paths,label,labels=gather_paths_all(jpg_path=data_path,num_classes=num_classes)
  count=len(label)
  for radius in radii:
    points=radius*4
    print("Calculating LBP with Radius: ", radius," Points: ", points," Count: ", count)
    batches=int(count/batch_size)+1
    f=np.zeros((count,points+2),float)
    for b in range(batches):
      st=b*batch_size
      end=(b+1)*batch_size
      if(end>count):
        end=count
      if(st==end):
        break
      images=dataset.gather_images_from_paths(paths,st,end-st)
      for i in range(end-st):
        _,f[st+i]=get_lbp_single(image=images[i],radius=radius)
    df=pd.DataFrame(f)
    Yl=[dataset.label_map[l-1] for l in label]
    #df = df.assign('0'=label)
    df = df.assign('0'=Yl)
    files=[f.split("/")[-1] for f in paths]
    df = df.assign(img=files)
    df=df.set_index('img')
    df.to_csv(storage_path+"_"+str(radius)+".csv")