source activate /home/ubuntu/StyleDiffusion/.conda
rm -rf precomputed
rm -rf checkpoint
python main.py --style_image gogh.jpg --dir_loss 6 --l1_loss_w 10.0 --style_loss_w 1.0 --removal_mode  diffusion
rm -rf precomputed
rm -rf checkpoint
python main.py --style_image gogh.jpg --dir_loss 6 --l1_loss_w 10.0 --style_loss_w 1.0 --removal_mode gaussian
rm -rf precomputed
rm -rf checkpoint
