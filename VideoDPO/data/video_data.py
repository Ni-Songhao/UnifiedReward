"""
modified from @author xiaowei MACVID dataset 
"""

import os
import random
import json
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import glob
import pandas as pd
import yaml


class TextVideo(Dataset):
    """
    Data is structured as follows.
        |video_dataset_0
            |clip1.mp4
            |clip2.mp4
            |...
            |metadata.json
    """

    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
    ):
        self.data_root = data_root
        self.resolution = resolution
        self.video_length = video_length
        self.subset_split = subset_split
        self.frame_stride = frame_stride
        self.clip_length = clip_length
        assert self.subset_split in ["train", "test", "all"]
        self.exts = ["avi", "mp4", "webm"]

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution, self.resolution]
        # assert(isinstance(self.resolution, list) and len(self.resolution) == 2)

        self._make_dataset()

    def _make_dataset(self):
        with open(self.data_root, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        self.videos = []
        for meta_path in self.config["META"]:
            metadata_path = os.path.join(meta_path, "dpo_data.json")
            with open(metadata_path, "r") as f:
                data = json.load(f)
                for item in data:
                    item['chosen'] = os.path.join(meta_path, '/'.join(item['chosen'].split('/')[-2:]))
                    item['rejected'] = os.path.join(meta_path, '/'.join(item['rejected'].split('/')[-2:]))

                    self.videos.append(item)


        print(f"Number of videos = {len(self.videos)}")

    def __getitem__(self, index):
        video = self.videos[index]

        video_list = []
        for path in [video['chosen'], video['rejected']]:
            video_reader = VideoReader(
                path,
                ctx=cpu(0),
                width=self.resolution[1],
                height=self.resolution[0],
            )

            all_frames = list(range(0, len(video_reader), self.frame_stride))
            if len(all_frames) < self.video_length:
                all_frames = list(range(0, len(video_reader), 1))

            # select random clip
            rand_idx = random.randint(0, len(all_frames) - self.video_length)
            frame_indices = all_frames[rand_idx : rand_idx + self.video_length]

            frames = video_reader.get_batch(frame_indices)
            assert (
                frames.shape[0] == self.video_length
            ), f"{len(frames)}, self.video_length={self.video_length}"

            frames = (
                torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
            )  # [t,h,w,c] -> [c,t,h,w]
            assert (
                frames.shape[2] == self.resolution[0]
                and frames.shape[3] == self.resolution[1]
            ), f"frames={frames.shape}, self.resolution={self.resolution}"
            frames = (frames / 255 - 0.5) * 2
            video_list.append(frames)


        data = {
            "chosen": video_list[0],
            "rejected": video_list[1],
            "caption": self.videos[index]['caption'],
        }
        return data

    def __len__(self):
        return len(self.videos)


"""
    A tipical item for DPO contain: [video1,video2,label0,caption]
    For better scalbility: we add a dataframe to choose video and leave the metadata.json still. 
"""


class TextVideoDPO(Dataset):
    """
    Data is structured as follows.
        |video_dataset_0
            |clip1.mp4
            |clip2.mp4
            |...
            |metadata.json
    """

    def __init__(
        self,
        data_root,
        resolution,
        video_length,
        frame_stride=4,
        subset_split="all",
        clip_length=1.0,
        dupbeta=1.0, # scale up factor
    ):
        self.data = TextVideo(
            data_root, resolution, video_length, frame_stride, subset_split, clip_length
        )
        self.pairs = []
        self.data_root = data_root
        with open(self.data_root, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        label_key = "label"
        self.dupbeta = dupbeta
        for meta_path in self.config["META"]:
            pairdata_path = os.path.join(meta_path, "dpo_data.json")
            with open(pairdata_path, "r") as f:
                pairs = json.load(f)
                for item in pairs:
                    # under the pair.json after 0601,label_key has no use
                    if dupbeta:
                        if 'score' not in item:
                            item['score']= 1

                    self.pairs.append(item)


        print(f"DPO dataset has {self.__len__()} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        if self.dupbeta:
            caption, prob_score = self.pairs[index]['caption'], self.pairs[index]['score']
            dupfactor = (0.72 / prob_score )**self.dupbeta # scale up factor 
        else:
            caption = self.pairs[index]['caption']

        videow = self.data[index]["chosen"]
        videol = self.data[index]["rejected"]
        if videow.dim()==5:
            combined_frames = torch.cat([videow, videol], dim=0)
        else:
            combined_frames = torch.cat([videow, videol], dim=0)
        if isinstance(caption, list):
            caption = caption[0]

        
        # print("in dataloader getitem",combined_frames.shape)
        if self.dupbeta:
            return {"video": combined_frames, "caption": caption,"dupfactor":dupfactor}
        else:
            return {"video": combined_frames, "caption": caption,"dupfactor":1.0}
