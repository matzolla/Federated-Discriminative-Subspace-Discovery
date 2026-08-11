# Extracting embeddings 

import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import torch
import json
import random
from tqdm import tqdm

class UCF101ClipDataset(Dataset):
    def __init__(self, root_dir, clip_length=16,num_clips=1,transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform
        self.num_clips=num_clips
        self.samples = self._make_dataset()
        self.classes = os.listdir(self.root_dir)
        self.classes.sort()

    # Create a mapping from class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    # To store frame indices used for training
        self.frame_indices_log = {}
    def _make_dataset(self):
        samples = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for video in os.listdir(class_path):
                    video_path = os.path.join(class_path, video)
                    samples.append((video_path, class_dir))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, class_label = self.samples[idx]
        video_clip=[]
        frames,frame_indices = self._load_video_clips(video_path)
        # Log frame indices for this video
        video_name = os.path.basename(video_path)
        self.frame_indices_log[video_name] = frame_indices
        # extend video_clip to get a long clip of self_num_clip*self.clip_length
        video =[video_clip.extend(clip) for clip in frames]
        #print(len(video_clip)) 
        return torch.stack(video_clip).permute(1,0,2,3), int(self.class_to_idx[class_label])

    def _load_video_clips(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        frame_indices = []  # To keep track of the indices of the frames
        index=0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self.transform(frame))
            frame_indices.append(index)
            index+=1
        cap.release()
        
        clips,clip_indices = self._generate_clips(frames,frame_indices)
        return clips,clip_indices

    def _generate_clips(self, frames,frame_indices):
        clips = []
        clip_indices_list=[]
        total_frames = len(frames)
        for start in range(0, total_frames, self.clip_length):
            end = start + self.clip_length
            if end <= total_frames:
                clip = frames[start:end]
                clip_indices = frame_indices[start:end]
                # lets also account for inconsistency in length of frames
                if len(clip)<self.clip_length:
                # we pad the clip with the last frame
                    pad = [clip[-1]] * (self.clip_length - len(clip))
                # extend clip to reach clip length
                    clip.extend(pad)
                    clip_indices.extend([clip_indices[-1]] * (self.clip_length - len(clip_indices)))
                clips.append(clip)
                clip_indices_list.append(clip_indices)
                
        # I want to select n-clips (non-overlapping) each of k-frames (both n and k are hyper-parameters)
        if len(clips)>= self.num_clips:
            selected_clips = random.sample(list(zip(clips, clip_indices_list)), self.num_clips)
            #return random.sample(clips,self.num_clips)
        else:
            # if the number of clips is <= k we can append the clips with the last clip
            selected_clips = list(zip(clips, clip_indices_list))
            while len(selected_clips)<self.num_clips:
                selected_clips.append(selected_clips[-1])
        selected_clips, selected_indices = zip(*selected_clips)
        return list(selected_clips) , list(selected_indices)

def get_dataloader(data_path, batch_size, clip_length=16,num_clips=1, num_workers=4):
    data_transform = transforms.Compose([
                                               transforms.ToPILImage(),
                                               transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                       ])
    dataset = UCF101ClipDataset(root_dir=data_path, clip_length=clip_length,num_clips=num_clips,transform=data_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),dataset