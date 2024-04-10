## Paper and implementation.

This is implementation of the StyleDiffusion [paper](https://arxiv.org/abs/2308.07863). The implementation is based on [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP/blob/master/diffusionclip.py)



## Instalation


```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```


## Run

In order to run code use `main.py`. Main parameters are `dir_loss, l1_loss_w, style_loss_w` which control loss weights and also `style_image` which is the path to 
style image.


Example of run:

```
python main.py --style_image munch.jpg --dir_loss 0.1 --l1_loss_w 0.1 --style_loss_w 5
```




