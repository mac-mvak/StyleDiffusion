f = open('runner.sh', 'w')
print('source activate /home/ubuntu/StyleDiffusion/.conda', file=f)
loss_l1 = [0.1, 0.25, 0.5, 0.75, 1.25, 2]
loss_dir = [5,6,7,8,9]
images = ['munch.jpg', 'gogh.jpg'] #, 'picasso.jpg']

default_loss = {
    'dir_loss': 1.,
    'l1_loss': 10.,
    'st_loss': 1.
}

image = 'picasso.jpg'

dir_loss = default_loss['dir_loss']
l1_loss = default_loss['l1_loss']
st_loss = default_loss['st_loss']

for new_loss in loss_l1:
    l1_loss = new_loss
    running_line = f"python main.py --style_image {image} --dir_loss {dir_loss} --l1_loss_w {l1_loss} --style_loss_w {st_loss}"
    print(running_line, file=f)

dir_loss = default_loss['dir_loss']
l1_loss = default_loss['l1_loss']
st_loss = default_loss['st_loss']

for new_loss in loss_dir:
    dir_loss = new_loss
    running_line = f"python main.py --style_image {image} --dir_loss {dir_loss} --l1_loss_w {l1_loss} --style_loss_w {st_loss}"
    print(running_line, file=f)



f.close()
