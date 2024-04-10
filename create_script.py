f = open('runner.sh', 'w')
print('source activate /home/ubuntu/StyleDiffusion/.conda', file=f)
loss_w = [0.1, 1, 5, 10]
images = ['munch.jpg', 'gogh.jpg'] #, 'picasso.jpg']

for image in images:
    for st_loss in loss_w:
        for dir_loss in loss_w:
            for l1_loss in loss_w:
                running_line = f"python main.py --style_image {image} --dir_loss {dir_loss} --l1_loss_w {l1_loss} --style_loss_w {st_loss}"
                print(running_line, file=f)

f.close()
