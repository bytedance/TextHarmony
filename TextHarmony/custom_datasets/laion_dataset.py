import os
import json
import random
from re import template
import numpy as np
from PIL import Image

from .loader import BaseDataset
from functools import cached_property


templates = {
    'TextToImage-Laion': [
        'Generate an image according to the caption.',
        'Produce an image in accordance with the specified description.',
        'Generate a visual representation based on the provided caption.',
        'Craft an illustration that reflects the given caption.',
        'Create an image that aligns with the given description.',
        'Produce a picture according to the provided caption.',
        'Generate a visual depiction corresponding to the given caption.',
    ],

    'TextToImage-TextCaps': [
        'Generate an image according to the caption.',
        'Produce an image in accordance with the specified description.',
        'Generate a visual representation based on the provided caption.',
        'Craft an illustration that reflects the given caption.',
        'Create an image that aligns with the given description.',
        'Produce a picture according to the provided caption.',
        'Generate a visual depiction corresponding to the given caption.',
    ],

    'SuperResolution': [
        'Make the text in this image more clear.',
        'Enhance the image\'s textual legibility.',
        'Improve the clarity of the text overlaying the image.',
        'Make the text within the visual more easily readable.',
        'Augment the visibility of the textual content in the picture.',
        'Optimize the readability of the textual elements in the image.',
        'Increase the sharpness of the text superimposed on the visual.',
        'Maximize the discernibility of the written information in the picture.',
        'Improve the definition of the textual components in the image.',
        'Enhance the intelligibility of the written text accompanying the visual.',
        'Amplify the clearness of the textual elements in the image.',
    ],

    'TextRemoval': [
        'Remove the text in this image.',
        'Remove the caption within this visual.',
        'Erase the written message from this picture.',
        'Eliminate the textual overlay in this image.',
        'Delete the words superimposed on this illustration.',
        'Clear the text content from this graphic.',
        'Omit the written elements in this artwork.',
        'Eradicate the textual inscription from this visual.',
        'Take out the caption overlaid on this picture.',
        'Wipe out the written communication from this image.',
        'Efface the textual components within this depiction.',
    ],

    'TextSeg': [
        'Extract the segmentation map of the text in this image.',
        'Extract the text\'s segmentation map from this visual.',
        'Isolate the segmentation map of the textual elements in this picture.',
        'Retrieve the segmentation map specifically for the written content in this artwork.',
        'Separate and extract the segmentation map corresponding to the text overlay in this image.',
        'Obtain the segmentation map that represents the boundaries of the written text in this illustration.',
        'Capture the segmentation map of the text regions within this graphic.',
        'Generate the segmentation map specifically targeting the textual components in this image.',
        'Derive the segmentation map that highlights the text regions in this visual.',
        'Acquire the segmentation map that accurately delineates the textual information in this picture.',
        'Retrieve the segmentation map depicting the spatial distribution of the written text in this image.',
    ],
}


class LaionDataset(BaseDataset):
    def __init__(
        self,
        annt_path,
        transform,
        total_length=None,
        phase="training",
        collate_mode="generate_segm",
        add_eos="",
        num_img_token=32,
        add_soi_token=True,
        text_first=False,
        context_type="current",
    ):
        super().__init__()

        self.annt_path = annt_path
        self.transform = transform

        assert phase in ["training", "validation"]
        self.phase = phase

        assert collate_mode in ["train", "generate_segm"]
        self.collate_mode = collate_mode
        self.add_eos = add_eos
        self.text_first = text_first

        assert context_type in [
            "multi_modal",
            "image_only",
            "text_only",
        ]
        self.context_type = context_type

        self.num_img_token = num_img_token
        self.add_soi_token = add_soi_token

        self.image_subseq = "<|image|>" * self.num_img_token
        if self.add_soi_token:
            self.image_subseq = "<|beginofimage|>" + self.image_subseq

        annt_file = annt_path
        self.annt_file = annt_file
        self.load_database()

        if total_length is not None:
            self.annts = self.annts[:total_length]

        print(f"length of the dataset is {len(self.annts)}")

    def load_database(self):
        with open(self.annt_file, "r") as rf:
            self.annts = json.load(rf)

    def __repr__(self) -> str:
        return (
            f"Segment Dataset phase={self.phase}\n"
            f"annotation_root={self.annt_path}\n"
            f"transform={self.transform}"
        )

    def __len__(self):
        return len(self.annts)

    def _get_image(self, image_path, return_image_path=False):
        try:
            image = self.loader(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(e)
            print(image_path)
            image = None

        if return_image_path:
            return image, image_path
        return image

    def _get_seg(self, seg_path, return_image_path=False):
        try:
            seg = np.load(seg_path).astype(np.uint8)
            seg[seg>0] = 255
            seg = Image.fromarray(seg).convert('RGB')
            seg = self.transform(seg)
        except Exception as e:
            print(e)
            print(seg_path)
            seg = None

        if return_image_path:
            return seg, seg_path
        return seg

    def __getitem__(self, index):
        item = self.annts[index]
        # meta = [index]

        images_tensor = []
        text = ""
        if self.collate_mode == "train":
            assert self.phase == "training"

            task_name = item['task']
            image_path = item['image']
            target_image_path = item['target_image']
            assert target_image_path is not None

            # caption = random.choice(templates[task_name])

            # if image_path is None:
            #     image_path = '/all_zero.jpg'

            # seg_path = os.path.join(item['anno_root'], 'charseg.npy')
            # annt, _ = self._get_image(image_path)
            image, image_dec = self._get_image(target_image_path)

            # if np.random.random() < 0.5:
            #     image = np.ascontiguousarray(image[:, ::-1])
            #     annt = np.ascontiguousarray(annt[:, ::-1])


            # if task_name == 'TextToImage-Laion' or task_name == 'TextToImage-TextCaps':
                # caption = caption + item['info']
            caption = item['info']  # !!!

            if self.text_first:
                text += f"{caption}{self.image_subseq}"
            else:
                text += f"{caption}{self.image_subseq}"
            
            # print(text)

            # images_tensor.append((annt, image_dec))
            images_tensor.append((image, image_dec))

        else:
            assert self.phase != "train"
            assert self.collate_mode == "generate_segm"

            annt = self._get_image(item["image"])

            if self.text_first:
                text += f"{item['caption']}{self.image_subseq}"
            else:
                text += f"{self.image_subseq}{item['caption']}"

            images_tensor.append(annt)

            # prepare target

            image = self._get_image(item["target_image"])
            text += self.image_subseq
            images_tensor.append(image)

        # meta.append(item["caption"])
        meta = {
            "index": index,
            "caption": caption,
            "dataset_name": "laion",
        }
        text = text.strip()
        if self.add_eos:
            text += self.add_eos

        return dict(text=text, images_tensor=images_tensor, meta=meta)

    # def __getitem__(self, index):
    #     item = self.annts[index]
    #     # meta = [index]

    #     images_tensor = []
    #     text = ""
    #     if self.collate_mode == "train":
    #         assert self.phase == "training"

    #         task_name = item['task']
    #         image_path = item['image']
    #         target_image_path = item['target_image']
    #         assert target_image_path is not None

    #         caption = random.choice(templates[task_name])

    #         if image_path is None:
    #             image_path = '/all_zero.jpg'

    #         # seg_path = os.path.join(item['anno_root'], 'charseg.npy')
    #         annt, _ = self._get_image(image_path)
    #         image, image_dec = self._get_image(target_image_path)

    #         if np.random.random() < 0.5:
    #             image = np.ascontiguousarray(image[:, ::-1])
    #             annt = np.ascontiguousarray(annt[:, ::-1])


    #         if task_name == 'TextToImage-Laion' or task_name == 'TextToImage-TextCaps':
    #             # caption = caption + item['info']
    #             caption = item['info']  # !!!

    #         if self.text_first:
    #             text += f"{caption}{self.image_subseq}{self.image_subseq}"
    #         else:
    #             text += f"{self.image_subseq}{caption}{self.image_subseq}"
            
    #         # print(text)

    #         images_tensor.append((annt, image_dec))
    #         images_tensor.append((image, image_dec))

    #     else:
    #         assert self.phase != "train"
    #         assert self.collate_mode == "generate_segm"

    #         annt = self._get_image(item["image"])

    #         if self.text_first:
    #             text += f"{item['caption']}{self.image_subseq}"
    #         else:
    #             text += f"{self.image_subseq}{item['caption']}"

    #         images_tensor.append(annt)

    #         # prepare target

    #         image = self._get_image(item["target_image"])
    #         text += self.image_subseq
    #         images_tensor.append(image)

    #     # meta.append(item["caption"])
    #     meta = {
    #         "index": index,
    #         "caption": caption,
    #         "dataset_name": "laion",
    #     }
    #     text = text.strip()
    #     if self.add_eos:
    #         text += self.add_eos

    #     return dict(text=text, images_tensor=images_tensor, meta=meta)

    @property
    def task_prefix(self):
        return f"_{self.context_type}"

    # def image_id_to_path(self, idx):
    #     image_id = self.annts[idx]["image_id"]

    #     image_path = os.path.join(
    #         self.data_root, "images", self.phase, f"{image_id}.jpg"
    #     )
    #     return image_path

    # def gt_id_to_path(self, idx):
    #     image_id = self.annts[idx]["image_id"]

    #     image_path = os.path.join(
    #         self.data_root, "annotations", self.phase, f"{image_id}.png"
    #     )
    #     return image_path

    @cached_property
    def palette(self):
        return [
            0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,
            3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,
            5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,
            255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,
            6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,
            92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,
            10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,
            0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,
            163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,
            0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,
            200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,
            163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,
            255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,
            255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,
            255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,
            122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,
            255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,
            255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,
            0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,
            0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,
            20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,
            255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,
            255,214,0,25,194,194,102,255,0,92,0,255
        ]



if __name__ == "__main__":
    from .utils import create_transform

    transform = create_transform(
        aug_type="flip", resolution=256, random_crop=False, random_flip=True
    )

    dataset = LaionDataset(
        data_root="./asset/ade20k/ADEChallengeData2016/",
        annt_root="./asset/ade20k/ADEChallengeData2016/",
        transform=transform,
        phase="training",
        collate_mode="generate_images",
        num_img_token=32,
        add_soi_token=True,
        context_type="multi_modal",
    )
    print(dataset)

