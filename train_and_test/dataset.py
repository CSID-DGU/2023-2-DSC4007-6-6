
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from config import Config

class VideoDataset(Dataset):
    def __init__(self,df,classes,augments=None,is_test=False):
        self.df=df
        self.augments=augments
        self.is_test=is_test
        self.video_names=self.df.video_name.unique()
        self.classes=classes
        self.num_classes=Config['NUM_CLASSES']
        
    def __getitem__(self,idx):
        video_name=self.video_names[idx]
        
        video_data=self.df.query(f'video_name=="{video_name}"')
        
        label_name=video_data['label'].iloc[0]
        
        # for cross-entropy
        #label=self.classes[label_name]
        
        #for BCEWithLogitsLoss
        label_index=self.classes[label_name] # encoded label 
        label=torch.zeros(self.num_classes)
        
        label[label_index]=1
        paths = video_data['paths'].tolist()
        paths=['.'+path for path in paths]
        
        # 샘플링
        paths=self._getpaths_(paths)
        
        frames=[]
        for path in paths:
            img=cv2.imread(path)
            img=cv2.resize(img,Config['IMG_SIZE'])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #img_squeezed=np.squeeze(img,axis=0)
            frames.append(img)
           # print(frames)
            
       # frames_tr=np.stack(frames,axis=2)

        if self.augments:
            augmented_frames=[]
            for frame in frames:
                augmented_frame=self.augments(image=frame)['image']
                #augmented_frame_squeezed=np.squeeze(augmented_frame,axis=0)
                augmented_frames.append(augmented_frame)
            frames_np=np.stack(augmented_frames,axis=2)
            frames_tr=torch.from_numpy(frames_np)
        else:
            frames_np=np.stack(frames,axis=2)
            frames_tr=torch.from_numpy(frames_np)
            
                
        frames_tr=np.squeeze(frames_tr,axis=0)
        if self.is_test:
            return frames_tr
        
        else:
            # label 추가 
            return frames_tr,label
            
        
    def _getpaths_(self,paths):
        num_frames=len(paths)
        
        if num_frames>Config['MAX_FRAMES']:
            indices = np.linspace(0, num_frames-1, Config['MAX_FRAMES'], dtype=int)
            sampled_paths = [paths[i] for i in indices]
        else:
            sampled_paths=paths
        
        
        return sampled_paths
    
    def __len__(self):
        return len(self.video_names)