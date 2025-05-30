import os
from PIL import Image
from itertools import product

def make_img_grid(images, rows=2, cols=2):
    if len(images) == 1:
        return images
    w, h = images[0].size
    if rows == 1 and cols == 1:
        return images
    else:     
        grid = Image.new('RGB', size=(cols*w, rows*h))

        for i, img in enumerate(images):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return [grid]

def save_image(images, image_path='', img_name=''):
    num_of_imgs = int(len(os.listdir(image_path)))
    if len(images) == 1:
        image = images[0]
        path = f'{image_path}/{img_name}_{num_of_imgs}.png'
        image.save(
            os.path.join(os.getcwd(), path)
        )
    else:
        for i, image in enumerate(images):
            path = f'{image_path}/{img_name}_{num_of_imgs + i}.png'
            image.save(
                os.path.join(os.getcwd(), path)
            )

def image_splitter(filename, save_dir):
    ''' Defaults '''
    sheet_w  = 1024
    sheet_h  = 1024
    base_col = 3
    base_row = 3 
    base_wh  = 340 
    base_gap = 3 

    name, ext = os.path.splitext(filename)
    img = Image.open(filename) 
    w, h = img.size
   
    scale = float(w) / sheet_w
    img_size = int(scale * base_wh)
    gap_size = int(scale * base_gap)

    grid = product(range(0, h - (h % img_size), img_size), range(0, w - (w % img_size), img_size))

    for r in range(base_row):
        start_y = (r * img_size) + (gap_size * (r + 1))
        for c in range(base_col):
            start_x = (c * img_size) + (gap_size * (c + 1))
            box = (start_x, start_y, start_x + img_size, start_y + img_size)
            out = os.path.join(save_dir, f'{name}_{r}_{c}{ext}')
            print('1')
            img.crop(box).save(out)


if __name__ == "__main__":
    image_splitter("/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/lora_images/cinderella/ComfyUI_00009_.png", 
                   "/home/naver/Documents/HieuDM/PycharmProjects/manga-generation-diffusion/lora_images/cinderella")