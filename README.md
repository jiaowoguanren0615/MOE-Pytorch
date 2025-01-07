<h1 align='center'>VisionTransFormer-MoE</h1>

## This project is implemented in PyTorch, can be used to train your image-datasets for vision tasks.  
## [Deepspeekv3 MoE source code](https://github.com/deepseek-ai/deepseek-v3)  
![image](https://github.com/jiaowoguanren0615/MOE-Pytorch/blob/main/sample_png/1.jpg)  
![image](https://github.com/jiaowoguanren0615/MOE-Pytorch/blob/main/sample_png/2.jpg)  


## Preparation

### Create conda virtual-environment
```bash
conda env create -f environment.yml
```

### Download the dataset: 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: VisionTransFormer-MoE Model
    ├── moe.py: Construct VisionTransFormer-MoE models
    ├── moe-kernal.py: Construct MoE modules
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── optimizer.py: Define Sophia/MARS optimizer
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  
Moreover, you can set the ___opt_auc___ parameter to True if you want to optimize your model for a better performance(maybe~).  

## Use MARS Optimizer (in util/optimizer.py)
You can use anther optimizer MARS, just need to change the optimizer in ___train_gpu.py___, for this training sample, can achieve better results
```
# optimizer = create_optimizer(args, model_without_ddp)
optimizer = MARS(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, lr_1d=args.adamw_lr)```

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```

### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error. 

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## ONNX Deployment
### step 1: ONNX export (modify the param of ___output___, ___model___ and ___checkpoint___)  
```bash
python onnx_export.py --model=vit_moe_base --output=./vit_moe_base.onnx --checkpoint=./output/vit_moe_base_best_checkpoint.pth
```

### step2: ONNX optimise
```bash
python onnx_optimise.py --model=vit_moe_base --output=./vit_moe_base_optim.onnx'
```

### step3: ONNX validate (modify the param of ___data_root___ and ___onnx-input___)  
```bash
python onnx_validate.py --data_root=/mnt/d/flower_data --onnx-input=./vit_moe_base_optim.onnx
```


## Citation
```
@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report}, 
      author={DeepSeek-AI and Aixin Liu and Bei Feng and Bing Xue and Bingxuan Wang and Bochao Wu and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and Chong Ruan and Damai Dai and Daya Guo and Dejian Yang and Deli Chen and Dongjie Ji and Erhang Li and Fangyun Lin and Fucong Dai and Fuli Luo and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Han Bao and Hanwei Xu and Haocheng Wang and Haowei Zhang and Honghui Ding and Huajian Xin and Huazuo Gao and Hui Li and Hui Qu and J. L. Cai and Jian Liang and Jianzhong Guo and Jiaqi Ni and Jiashi Li and Jiawei Wang and Jin Chen and Jingchang Chen and Jingyang Yuan and Junjie Qiu and Junlong Li and Junxiao Song and Kai Dong and Kai Hu and Kaige Gao and Kang Guan and Kexin Huang and Kuai Yu and Lean Wang and Lecong Zhang and Lei Xu and Leyi Xia and Liang Zhao and Litong Wang and Liyue Zhang and Meng Li and Miaojun Wang and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Mingming Li and Ning Tian and Panpan Huang and Peiyi Wang and Peng Zhang and Qiancheng Wang and Qihao Zhu and Qinyu Chen and Qiushi Du and R. J. Chen and R. L. Jin and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and Runxin Xu and Ruoyu Zhang and Ruyi Chen and S. S. Li and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shaoqing Wu and Shengfeng Ye and Shengfeng Ye and Shirong Ma and Shiyu Wang and Shuang Zhou and Shuiping Yu and Shunfeng Zhou and Shuting Pan and T. Wang and Tao Yun and Tian Pei and Tianyu Sun and W. L. Xiao and Wangding Zeng and Wanjia Zhao and Wei An and Wen Liu and Wenfeng Liang and Wenjun Gao and Wenqin Yu and Wentao Zhang and X. Q. Li and Xiangyue Jin and Xianzu Wang and Xiao Bi and Xiaodong Liu and Xiaohan Wang and Xiaojin Shen and Xiaokang Chen and Xiaokang Zhang and Xiaosha Chen and Xiaotao Nie and Xiaowen Sun and Xiaoxiang Wang and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xingkai Yu and Xinnan Song and Xinxia Shan and Xinyi Zhou and Xinyu Yang and Xinyuan Li and Xuecheng Su and Xuheng Lin and Y. K. Li and Y. Q. Wang and Y. X. Wei and Y. X. Zhu and Yang Zhang and Yanhong Xu and Yanhong Xu and Yanping Huang and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Li and Yaohui Wang and Yi Yu and Yi Zheng and Yichao Zhang and Yifan Shi and Yiliang Xiong and Ying He and Ying Tang and Yishi Piao and Yisong Wang and Yixuan Tan and Yiyang Ma and Yiyuan Liu and Yongqiang Guo and Yu Wu and Yuan Ou and Yuchen Zhu and Yuduan Wang and Yue Gong and Yuheng Zou and Yujia He and Yukun Zha and Yunfan Xiong and Yunxian Ma and Yuting Yan and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Z. F. Wu and Z. Z. Ren and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhen Huang and Zhen Zhang and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhibin Gou and Zhicheng Ma and Zhigang Yan and Zhihong Shao and Zhipeng Xu and Zhiyu Wu and Zhongyu Zhang and Zhuoshu Li and Zihui Gu and Zijia Zhu and Zijun Liu and Zilin Li and Ziwei Xie and Ziyang Song and Ziyi Gao and Zizheng Pan},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437}, 
}
```