from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

def make_caption_tile(img_pil, caption, tile_size=224, caption_height=40):
    """
    Returns a tile of size (tile_size, tile_size + caption_height)
    with the image centered on top and the caption centered below.
    """
    # Ensure image is exactly tile_size x tile_size for consistent layout
    img_resized = img_pil.resize((tile_size, tile_size), Image.BILINEAR)

    tile = Image.new("RGB", (tile_size, tile_size + caption_height), color=(255, 255, 255))
    tile.paste(img_resized, (0, 0))

    draw = ImageDraw.Draw(tile)
    # Try a default font; if you want custom fonts, set font_path to a .ttf file
    try:
        font = ImageFont.load_default()
    except:
        font = None

    text_w, text_h = draw.textbbox((0, 0), caption, font=font)[2:]
    x = (tile_size - text_w) // 2
    y = tile_size + (caption_height - text_h) // 2
    draw.text((x, y), caption, fill=(0, 0, 0), font=font)
    return tile

def concat_tiles_horizontally(tiles, spacing=8, bg_color=(245, 245, 245)):
    """
    Concatenate PIL images of the same height horizontally with spacing.
    """
    widths = [t.width for t in tiles]
    heights = [t.height for t in tiles]
    assert len(set(heights)) == 1, "All tiles must have the same height."
    h = heights[0]
    total_w = sum(widths) + spacing * (len(tiles) - 1)

    out = Image.new("RGB", (total_w, h), color=bg_color)
    x = 0
    for i, tile in enumerate(tiles):
        out.paste(tile, (x, 0))
        x += tile.width + (spacing if i < len(tiles) - 1 else 0)
    return out

def build_augmentations(input_size=224):
    """
    Define five separate augmentation ops. Each will be applied independently.
    Names are user-friendly for captions.
    """
    aug_ops = [
        ("Horizontal Flip",
         transforms.Compose([
             transforms.Resize((input_size, input_size)),
             transforms.RandomHorizontalFlip(p=1.0)  # always flip
         ])),
        ("Zoom (ResizedCrop 0.9–1.0)",
         transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0))),
        ("Rotate (±25°)",
         transforms.Compose([
             transforms.Resize((input_size, input_size)),
             transforms.RandomAffine(degrees=25, fill=0)
         ])),
        ("Translate (±10%)",
         transforms.Compose([
             transforms.Resize((input_size, input_size)),
             transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0)
         ])),
        ("Shear (20°)",
         transforms.Compose([
             transforms.Resize((input_size, input_size)),
             transforms.RandomAffine(degrees=0, shear=20, fill=0)
         ])),
    ]
    return aug_ops

def apply_and_combine_per_image(
    image_path,
    output_dir="augmented_panels",
    input_size=224,
    caption_height=40,
    spacing=8
):
    """
    For a single image:
      - Create tiles: Original + 5 augmentations
      - Write operation names under each result
      - Concatenate into one output image and save
    """
    os.makedirs(output_dir, exist_ok=True)

    base_img = Image.open(image_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Original tile
    original_tile = make_caption_tile(
        base_img.resize((input_size, input_size), Image.BILINEAR),
        "Original",
        tile_size=input_size,
        caption_height=caption_height
    )

    # Five separate augmentations
    aug_ops = build_augmentations(input_size=input_size)
    aug_tiles = []
    for caption, aug in aug_ops:
        aug_img = aug(base_img)
        # Some transforms (e.g., RandomResizedCrop) already output correct size,
        # but we ensure consistency anyway:
        aug_tiles.append(make_caption_tile(
            aug_img,
            caption,
            tile_size=input_size,
            caption_height=caption_height
        ))

    # Combine tiles horizontally: Original + 5 augmentations
    panel = concat_tiles_horizontally([original_tile] + aug_tiles, spacing=spacing)

    # Save
    save_path = os.path.join(output_dir, f"{base_name}_panel.jpg")
    panel.save(save_path, quality=95)
    print(f"✅ Saved panel: {save_path}")
    return save_path

def process_images(
    image_paths,
    output_dir="augmented_panels",
    input_size=224,
    caption_height=40,
    spacing=8
):
    """
    Apply the pipeline to each image path in the list.
    """
    saved = []
    for p in image_paths:
        saved.append(apply_and_combine_per_image(
            p,
            output_dir=output_dir,
            input_size=input_size,
            caption_height=caption_height,
            spacing=spacing
        ))
    return saved


images = ['image_eigencam/Plautia crossota 55.jpg',
               'image_eigencam/Acalymma vittatum 17.jpg',
               'image_eigencam/Brachytrupes portentosus 44.jpg',
               'image_eigencam/Cicadidae 18.jpg',
               'image_eigencam/Helicoverpa armigera 39.jpg']
process_images(images, output_dir="augmented_samples", input_size=256)



