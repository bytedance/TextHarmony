import os
import os.path as osp
import json
from PIL import Image
from datetime import datetime
import numpy as np
import torch
from torchvision.utils import save_image
import random

from evaluate_utils import evaluate_relaxed_accuracy, evaluateANLS, evaluate_exact_match_accuracy

from TextHarmony.models.utils.monkey_patch import (
    replace_llama_attn_with_flash_attn,
    replace_blip2_attn_with_qknorm_attn,
    replace_beam_search,
    replace_stable_diffusion_pipeline_call,
    replace_stable_diffusion_unet_forward,
)

replace_beam_search()
replace_blip2_attn_with_qknorm_attn()
replace_stable_diffusion_unet_forward()
replace_stable_diffusion_pipeline_call()
IS_TRAIN = False
if IS_TRAIN:
    replace_llama_attn_with_flash_attn()


from TextHarmony.models import TextHarmony
from TextHarmony.custom_datasets.utils import create_transform
from TextHarmony.custom_datasets.wds_utils import init_tokenizer
from TextHarmony.utils import (
    ArgumentParser,
    TrainingArguments,
    init_distributed_mode,
    load_model_weights,
)
from TextHarmony.utils.clip_sim_score import tensor_to_pil

import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


def model_gen(model, image_paths, question,
                transform,
                tokenizer,
                num_total_token=2048,
                truncation=True,
                num_img_token=512,
                generation_kwargs=None,
            ):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)
    image_tensors = np.stack(images, axis=0)

    image_subseq = "<|beginofimage|>" + "<|image|>" * num_img_token

    text = "Based on the image, please answer the question. {image}{question} The answer is:".format(image=image_subseq, question=question)

    text = (
            text.replace("<|image|> ", "<|image|>")
            .replace(" <|image|>", "<|image|>")
            .replace(" <|beginofimage|>", "<|beginofimage|>")
            .replace("<|beginofimage|> ", "<|beginofimage|>")
        )

    # print(text)
    
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=num_total_token,
        truncation=truncation,
        padding="do_not_pad",
        return_tensors="np",
        return_attention_mask=True,
    )
    text_ids = text_tensor["input_ids"][0]
    text_attn_mask = text_tensor["attention_mask"][0]

    # print('text_ids: ', len(text_ids))

    image_tensors = torch.from_numpy(image_tensors)
    num_images = image_tensors.shape[0]
    target_image_idxs = torch.tensor([num_images - 1], dtype=torch.long)

    
    task_identifiers = [
        ["Generate an image", "Fill the masked"],
        [""]
    ]
    meta = {}
    meta["task_id"] = None
    for task_id, idents in enumerate(task_identifiers):
            flag = False
            for ident in idents:
                if ident.lower() in text.lower():
                    flag = True
                    break
            if flag:
                meta["task_id"] = task_id
                break
    assert meta["task_id"] is not None
    
    
    _data = dict(
        image_tensors=image_tensors,
        image_tensors_dec=None,
        text_ids=torch.from_numpy(text_ids)[None, ...],
        attention_mask=torch.from_numpy(text_attn_mask)[None, ...],
        num_image_per_seq=torch.tensor([num_images]),
        nearest_bos_idxs=None,
        meta=meta,
        target_image_idxs=target_image_idxs,
    )

    if generation_kwargs is not None:
        for k, v in generation_kwargs.items():
            _data[k] = v
    
    inputs = _data
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device="cuda")
            inputs[k] = v

    try:
    
        outputs = model.generate(mode="generate_texts", **inputs)

        generate_texts = tokenizer.batch_decode(
                        outputs["text_ids"], skip_special_tokens=True
                    )
    except:
        generate_texts = ['']
        print('wrong output')

    # print(question)
    # print(image_paths[0])
    # print(generate_texts)
    return generate_texts


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    ques = []
    for line in lines:
        ques.append(json.loads(line))
    return ques


class MyDataset(Dataset):

    def __init__(self, tot_len):
        super().__init__()
        self.tot_len = tot_len

    def __len__(self):
        return self.tot_len

    def __getitem__(self, index):
        return index


def main():
    setup_seed(32)
    parser = ArgumentParser(TrainingArguments)
    init_distributed_mode()
    args = parser.parse_args_with_config_file_into_dataclasses()
    train_args, config = args
    # print(train_args)
    # print(config)

    rank = dist.get_rank()
    pid = os.getpid()
    print(f'current pid: {pid}')
    print(f'Current rank {rank}')
    device_id = rank % torch.cuda.device_count()

    

    print("Model Init Start")
    model = TextHarmony(**config.model)
    if getattr(config, "load_from", None):
        load_model_weights(model, config.load_from, image_upscale=config.image_upscale)
    model = model.to(device="cuda:{}".format(device_id))

    model.eval()

    del model.image_decoder
    model.image_decoder = None


    ddp_model = DistributedDataParallel(model, device_ids=[device_id])
    dist.barrier()



    
    # model = model.to(device="cuda")
    # model.eval()
    ddp_model.eval()

    del model

    tokenizer = init_tokenizer(config.inference.tokenizer_path)
    transform = create_transform(**config.inference.transform)

    if 'jsonl' in config.data_path:
        data = load_jsonl(config.data_path)
    else:
        data = json.load(open(config.data_path))


    dataset = MyDataset(len(data))
    # dataset = MyDataset(100)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    torch.cuda.empty_cache()


    human_part = []
    # for i in range(100): # range(len(data)):
    
    from tqdm import tqdm
    for i in tqdm(dataloader):
        image = data[i]["image"] if "image" in data[i] else data[i]["image_path"]
        # image = 'data/chartqa/ChartQA Dataset/test/png/' + data[i]["imgname"] # !!!
        # question =  data[i]["query"]
        question = data[i]["question"] # 
        image_paths = [os.path.join(config.data_root, image)]
        response = model_gen(ddp_model.module, image_paths, question, transform, tokenizer, num_img_token=config.model.num_img_token, generation_kwargs=config.inference.generation_kwargs)[0]
        human_part.append({
            'answer': response,
            #'annotation': data[i]['label'],
            'annotation': data[i]['answer'] if 'answer' in data[i] else data[i]['answers'],
        }) 

    # 收集所有进程的 human_part 列表
    human_part_lists = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(human_part_lists, human_part)
    
    if dist.get_rank() == 0:

        # 连接所有进程的 human_part 列表
        full_human_part = []
        for part in human_part_lists:
            full_human_part.extend(part)

        human_part = full_human_part

        print('human_part: ', len(human_part))

        acc = evaluate_exact_match_accuracy(human_part)
        anls = evaluateANLS(human_part)

        print('acc: ', acc)
        print('anls: ', anls)

if __name__ == "__main__":
    with torch.no_grad():
        main()
