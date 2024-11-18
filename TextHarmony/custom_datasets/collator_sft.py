from sys import _current_frames
from tkinter.messagebox import NO
from typing import Any
import numpy as np

import torch

from .wds_utils import init_tokenizer


class MultiImageCollator:
    def __init__(
        self,
        tokenizer_path,
        mode="train",
        generation_kwargs=None,
        padding="longest",
        ignore_image_loss_idx=-1,
        num_img_token=32
    ):
        """
        Designed for VIST Dataset
        """

        self.tokenizer = init_tokenizer(tokenizer_path)
        self.mode = mode
        self.generation_kwargs = generation_kwargs
        self.padding = padding
        self.ignore_image_loss_idx = ignore_image_loss_idx

        self.num_img_token = num_img_token
        self.image_subseq = "<|image|>" * self.num_img_token
        self.image_subseq = "<|beginofimage|>" + self.image_subseq

        self.task_identifiers = [
            ["Generate an image", "Fill the masked"],
            [""]
        ]

    def set_mode(self, mode):
        self.mode = mode

    def __call__(self, data_list) -> Any:
        if self.mode == "train":
            return self._call_for_train(data_list)
        elif self.mode == "generate_texts":
            return self._call_for_generate_texts(data_list)
        elif self.mode == "generate_images":
            return self._call_for_generate_images(data_list)
        elif self.mode == "generate_both":
            raise NotImplementedError(
                f"Get {self.mode}, please specify the exact mode before calling it"
            )
        elif self.mode == "generate_segm":
            return self._call_for_generate_images(data_list)
        else:
            raise NotImplementedError(
                f"collate_mode {self.mode} is NOT supported by far"
            )

    def _call_for_generate_texts(self, data_list):
        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []
        meta = []
        text_inputs = []

        for data in data_list:
            cur_meta = data["meta"]

            images_tensor = data["images_tensor"]
            if len(images_tensor) > 0:
                images_tensor = self._convert_images_tensor(images_tensor)
                if isinstance(images_tensor, tuple):
                    images_tensor, images_tensor_dec = images_tensor
                    images_tensors_dec_all += images_tensor_dec
                images_tensors_all += images_tensor
                num_image_per_seq.append(len(images_tensor))

            text_inputs.append(data["text"])

            cur_meta["task_id"] = None

            for task_id, idents in enumerate(self.task_identifiers):
                flag = False
                for ident in idents:
                    if ident.lower() in data["text"].lower():
                        flag = True
                        break
                if flag:
                    cur_meta["task_id"] = task_id
                    break
            assert cur_meta["task_id"] is not None

            meta.append(cur_meta)

        self.tokenizer.padding_side = "left"
        text_tensor = self.tokenizer(
            text_inputs,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = None
        if len(images_tensors_all) > 0:
            images_tensors = torch.stack(images_tensors_all, dim=0)

        image_tensors_dec = None
        if len(images_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        if len(num_image_per_seq) > 0:
            num_image_per_seq = torch.tensor(
                num_image_per_seq, dtype=torch.long, device=images_tensors.device
            )
        else:
            num_image_per_seq = None

        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            meta=meta,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def _call_for_generate_images(self, data_list):
        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []
        meta = []
        text_inputs = []
        target_image_idxs = []

        for data in data_list:
            cur_meta = data["meta"]

            images_tensor = data["images_tensor"]
            assert len(images_tensor) > 0

            images_tensor = self._convert_images_tensor(images_tensor)
            if isinstance(images_tensor, tuple):
                images_tensor, images_tensor_dec = images_tensor
                images_tensors_dec_all += images_tensor_dec
            images_tensors_all += images_tensor
            num_image_per_seq.append(len(images_tensor))
            target_image_idxs.append(sum(num_image_per_seq) - 1)

            text_inputs.append(data["text"])

            cur_meta["task_id"] = None

            for task_id, idents in enumerate(self.task_identifiers):
                flag = False
                for ident in idents:
                    if ident.lower() in data["text"].lower():
                        flag = True
                        break
                if flag:
                    cur_meta["task_id"] = task_id
                    break
            assert cur_meta["task_id"] is not None

            meta.append(cur_meta)
        
        self.tokenizer.padding_side = "right"
        text_tensor = self.tokenizer(
            text_inputs,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        image_tensors_dec = None
        if len(images_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )
        target_image_idxs = torch.tensor(
            target_image_idxs, dtype=torch.long, device=images_tensors.device
        )

        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            meta=meta,
            target_image_idxs=target_image_idxs,
        )

        if self.generation_kwargs is not None:
            for k, v in self.generation_kwargs.items():
                data[k] = v

        return data

    def _get_vqa(self, question, answer):
        default_instr_prompts = [
            "The answer is:",
            "Based on the image, please answer the question. {image}{question}",
            "",
        ]
        assis_prompt, user_prompt, sys_prompt = default_instr_prompts
        text_input = user_prompt.format(
                    image=self.image_subseq, question=question
                )
        text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()
        ignore_prompt_token_offset = self.tokenizer(
            text_input.strip(), return_tensors="pt"
        ).attention_mask.sum(1)
        text_input += " " + answer + self.tokenizer.eos_token
        return text_input, ignore_prompt_token_offset

    def _get_generate_image(self, question, answer):
        default_instr_prompts = [
            "",
            "{image}{question}",
            "",
        ]
        assis_prompt, user_prompt, sys_prompt = default_instr_prompts
        text_input = user_prompt.format(
                    image=self.image_subseq, question=question
                )
        text_input = f"{sys_prompt} {text_input} {assis_prompt}".strip()
        ignore_prompt_token_offset = self.tokenizer(
            text_input.strip(), return_tensors="pt"
        ).attention_mask.sum(1)
        text_input += " " + answer + self.tokenizer.eos_token
        return text_input, ignore_prompt_token_offset
    
    def _call_for_train(self, data_list):
        images_tensors_all = []
        num_image_per_seq = []
        images_tensors_dec_all = []
        meta = []
        text_inputs = []
        image_loss_mask_all = []

        ignore_prompt_token_offsets = []

        for data in data_list:

            cur_meta = data['meta']

            images_tensor = data["images_tensor"]
            assert len(images_tensor) > 0
            
            is_generate_image = data.get("is_generate_image", False)
            question, answer = data["question"], data["answer"]

            cur_meta["task_id"] = None

            for task_id, idents in enumerate(self.task_identifiers):
                flag = False
                for ident in idents:
                    if ident.lower() in question.lower():
                        flag = True
                        break
                if flag:
                    cur_meta["task_id"] = task_id
                    break
            assert cur_meta["task_id"] is not None

            if is_generate_image:
                text_input, ignore_prompt_token_offset = self._get_generate_image(question, answer)
            else:
                text_input, ignore_prompt_token_offset = self._get_vqa(question, answer)

            ignore_image_idx = data.get("ignore_image_idx", -1)

            images_tensor = self._convert_images_tensor(images_tensor)
            if isinstance(images_tensor, tuple):
                images_tensor, images_tensor_dec = images_tensor
                images_tensors_dec_all += images_tensor_dec
            images_tensors_all += images_tensor
            num_image_per_seq.append(len(images_tensor))

            image_loss_mask = [1.] * len(images_tensor)
            if self.ignore_image_loss_idx >= 0:
                image_loss_mask[self.ignore_image_loss_idx] = 0.
            if ignore_image_idx >= 0:
                image_loss_mask[ignore_image_idx] = 0.
            for cur_image_loss_mask in image_loss_mask:
                image_loss_mask_all.append(cur_image_loss_mask)

            text_inputs.append(text_input)
            ignore_prompt_token_offsets.append(ignore_prompt_token_offset)
            
            meta.append(cur_meta)

        self.tokenizer.padding_side = "right"

        text_tensor = self.tokenizer(
            text_inputs,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True,
        )
        text_ids = text_tensor.input_ids
        attn_mask = text_tensor.attention_mask

        images_tensors = torch.stack(images_tensors_all, dim=0)
        image_tensors_dec = None
        if len(images_tensors_dec_all) > 0:
            image_tensors_dec = torch.stack(images_tensors_dec_all, dim=0)
            assert image_tensors_dec.shape[0] == images_tensors.shape[0]

        image_loss_mask = None
        if len(image_loss_mask_all) > 0:
            image_loss_mask = torch.tensor(
                image_loss_mask_all, device=images_tensors.device
            )
            image_loss_mask = image_loss_mask.squeeze(-1)

        num_image_per_seq = torch.tensor(
            num_image_per_seq, dtype=torch.long, device=images_tensors.device
        )

        data = dict(
            image_tensors=images_tensors,
            image_tensors_dec=image_tensors_dec,
            num_image_per_seq=num_image_per_seq,
            text_ids=text_ids,
            attention_mask=attn_mask,
            meta=meta,
            image_loss_mask=image_loss_mask,
            ignore_prompt_token_offset=ignore_prompt_token_offsets,
        )

        return data

    def _convert_images_tensor(self, images_tensor):
        if isinstance(images_tensor[0], tuple):
            images_tensor_dec = [i[1] for i in images_tensor]
            images_tensor = [i[0] for i in images_tensor]
            map_fn = (
                torch.from_numpy
                if isinstance(images_tensor[0], np.ndarray)
                else lambda x: x
            )
            images_tensor = [map_fn(image_tensor) for image_tensor in images_tensor]
            images_tensor_dec = [
                map_fn(image_tensor) for image_tensor in images_tensor_dec
            ]
            return images_tensor, images_tensor_dec
        else:
            map_fn = (
                torch.from_numpy
                if isinstance(images_tensor[0], np.ndarray)
                else lambda x: x
            )
            images_tensor = [map_fn(image_tensor) for image_tensor in images_tensor]
            return images_tensor
