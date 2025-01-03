from PIL import Image
def resize_image(img, max_size=1024):
    width, height = img.size
    if width > height:
        ratio = max_size / width
    else:
        ratio = max_size / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
    return resized_img