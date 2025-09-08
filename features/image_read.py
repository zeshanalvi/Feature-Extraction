import cv2, numpy as np, os


class Dataset:
   def __init__(self,num_classes=16):
      self.num_classes=num_classes
      self.img_cols=224
      self.img_rows=224
      self.img_channels=3
      self.images=None
      self.labels=None
      self.label=None
      if(self.num_classes==8):
         self.label_map=['ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-pylorus', 'polyps']
      if(self.num_classes==16):
         self.label_map=['retroflex-rectum', 'out-of-patient', 'ulcerative-colitis', 'normal-cecum', 'normal-z-line', 'dyed-lifted-polyps', 'blurry-nothing', 'retroflex-stomach', 'instruments', 'dyed-resection-margins', 'stool-plenty', 'esophagitis', 'normal-pylorus', 'polyps', 'stool-inclusions', 'colon-clear']
      if(self.num_classes==23):
         self.label_map=['barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum', 'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d',
             'hemorrhoids', 'ileum', 'impacted-stool', 'normal-z-line', 'polyps', 'pylorus', 'retroflex-rectum', 'retroflex-stomach',
             'ulcerative-colitis-0-1', 'ulcerative-colitis-1-2', 'ulcerative-colitis-2-3', 'ulcerative-colitis-grade-1', 'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-3']
      if(self.num_classes==36):
         self.label_map=['barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3','cecum', 'normal-cecum', 'dyed-lifted-polyps', 'dyed-resection-margins',
         'esophagitis','esophagitis-a','esophagitis-b-d','hemorrhoids', 'ileum', 'impacted-stool','normal-z-line','polyps','pylorus','normal-pylorus',
         'retroflex-rectum','retroflex-stomach','ulcerative-colitis','ulcerative-colitis-0-1','ulcerative-colitis-1-2','ulcerative-colitis-2-3',
         'ulcerative-colitis-grade-1','ulcerative-colitis-grade-2','ulcerative-colitis-grade-3',
         'lesion', 'dysplasia', 'cancer', 'blurry-nothing', 'colon-clear', 'stool-inclusions', 'stool-plenty', 'instruments', 'out-of-patient']
   
   def gather_images_from_paths(self,jpg_paths,start,count):
      print('Stats of Images Start:',start,' To:',(start+count),'All Images:',len(jpg_paths))
      ima=np.zeros((count,self.img_rows,self.img_cols,self.img_channels),np.uint8)
      for i in range(count):
        img=cv2.imread(jpg_paths[start+i])
        im = cv2.resize(img, (self.img_rows, self.img_cols)).astype(np.uint8)
        ima[i]=im
      self.images=ima
      return ima
   
   def get_labels(self,labels):
      return [self.label_map[l-1] for l in labels]
   
   def paths_no_class(self,base_path,folder,count,ima,label):
      i=0
      for f in folder:
            im=base_path+f
            ima[i]=im
            label[i]=0
            i+=1
            if(count<i):
               break
      return ima, label
   def paths_classes(self,base_path,folder,count,ima,label):
      i=0
      for fldr in folder:
        for f in os.listdir(base_path+fldr+"/"):
            im=base_path+fldr+"/"+f
            ima[i]=im
            label[i]=self.label_map.index(fldr)+1
            i+=1
        if(count<=i):
            break
      return ima, label
   def gather_paths_all(self,jpg_path):
      folder=os.listdir(jpg_path)
      count=0
      if (os.path.isfile(jpg_path+folder[0])):
         count=len(os.listdir(jpg_path))
      else:
         count=sum([len(os.listdir(jpg_path+f)) for f in os.listdir(jpg_path)])
      ima=['' for x in range(count)]
      labels=np.zeros((count,self.num_classes),dtype=float)
      label=[0 for x in range(count)]
      if (os.path.isfile(jpg_path+folder[0])):
         ima,label=self.paths_no_class(jpg_path,folder,count,ima,label)
      else:
        ima,label=self.paths_classes(jpg_path,folder,count,ima,label)
      for i in range(count):
         labels[i][label[i]-1]=1
      return ima, label ,labels