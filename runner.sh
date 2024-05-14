source activate /home/ubuntu/StyleDiffusion/.conda
python main.py --style_image munch.jpg --dir_loss 1.0 --l1_loss_w 1.25 --style_loss_w 1.0
python main.py --style_image munch.jpg --dir_loss 6 --l1_loss_w 10.0 --style_loss_w 1.0
python main.py --style_image gogh.jpg --dir_loss 1.0 --l1_loss_w 1.25 --style_loss_w 1.0
python main.py --style_image gogh.jpg --dir_loss 6 --l1_loss_w 10.0 --style_loss_w 1.0
python main.py --style_image cezanne.jpg --dir_loss 1.0 --l1_loss_w 1.25 --style_loss_w 1.0
python main.py --style_image cezanne.jpg --dir_loss 6 --l1_loss_w 10.0 --style_loss_w 1.0
python main.py --style_image gogh.jpg --dir_loss 6.1 --gaussian_kernel 0.5 --l1_loss_w 10.0 --style_loss_w 1.0 --removal_mode gaussian