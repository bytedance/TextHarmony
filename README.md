# Harmonizing Visual Text Comprehension and Generation

## Environment

**step 1**: set up the environment

```
git clone https://github.com/bytedance/TextHarmony
cd TextHarmony
pip install -r requirements.txt
# install `MultiScaleDeformableAttention` module
cd TextHarmony/models/utils/ops
python setup.py install
```
some of the packages like mmcv and flash_attn in requirements.txt may need to be installed manually.

**step 2**: download pretraining weights

```
cd TextHarmony
python TextHarmony/scripts/download_hf_models.py
```

**step 3**: download the model weight of [TextHarmony](https://huggingface.co/jingqun/textharmony)

```
# concatenate the model files
cat pytorch_model.binaa pytorch_model.binab pytorch_model.binac > pytorch_model.bin
```

## Inference

**step1**: modify 'load_from', 'llm_model_path', 'encoder_model_path' and 'pretrained_model_name_or_path' in example_inference.yaml

**step 2**: run the following command:

```
torchrun --nproc_per_node 1 --nnodes 1 --master_port 2333 inference.py  --config_file=TextHarmony/TextHarmony/configs/release/example_inference.yaml
```

## Evaluation

### image comprehension

**step1**: modify 'data_root' and 'data_path' in 896-moe-eval.yaml. The structure of 'data_path' should be as follows:

```
[
    {
		"image": image_path,
		"question": question,
		"answer": answer
    },
]
```

**step 2**: run the following command

```
torchrun --nproc_per_node 1 --nnodes 1 --master_port 2333 evaluate.py --config_file=TextHarmony/TextHarmony/configs/release/896-moe-eval.yaml
```

### image generation

**step 1**: download [AnyText-Benchmark](https://github.com/tyxsspa/AnyText?tab=readme-ov-file)

**step 2**: generate the target images

```
torchrun --nproc_per_node 1 --nnodes 1 --master_port 2333 inference.py  --config_file=TextHarmony/TextHarmony/configs/release/896-moe-inference.yaml
```

**step 3**: calculate the results

```
python TextHarmony/image_eval/eval_dgocr.py
```

## Training

* **TODO**

## Acknowledgment

We thank the great work of [MM-Interleaved](https://github.com/OpenGVLab/MM-Interleaved), [TextDiffuser](https://github.com/microsoft/unilm/tree/master/textdiffuser-2), [AnyText](https://github.com/tyxsspa/AnyText) and [LoRAMoE](https://github.com/Ablustrund/LoRAMoE)

## Citation

```
@article{zhao2024harmonizing,
  title={Harmonizing Visual Text Comprehension and Generation},
  author={Zhao, Zhen and Tang, Jingqun and Wu, Binghong and Lin, Chunhui and Wei, Shu and Liu, Hao and Tan, Xin and Zhang, Zhizhong and Huang, Can and Xie, Yuan},
  journal={arXiv preprint arXiv:2407.16364},
  year={2024}
}
```
