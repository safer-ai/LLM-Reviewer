#!/usr/bin/env python3
"""
Create an icon for the Document Reviewer application
Creates both .ico (Windows) and .icns (Mac) formats
"""

from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path


def create_icon_image(size=256):
    """Create a simple icon for the document reviewer"""
    # Create a new image with a nice background
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw background circle
    margin = size // 8
    draw.ellipse([margin, margin, size-margin, size-margin], 
                 fill=(41, 128, 185, 255))  # Nice blue color
    
    # Draw document symbol
    doc_margin = size // 4
    doc_width = size // 2
    doc_height = int(doc_width * 1.3)
    doc_x = (size - doc_width) // 2
    doc_y = (size - doc_height) // 2
    
    # Document background
    draw.rectangle([doc_x, doc_y, doc_x + doc_width, doc_y + doc_height],
                   fill=(255, 255, 255, 255))
    
    # Document corner fold
    fold_size = doc_width // 4
    draw.polygon([
        (doc_x + doc_width - fold_size, doc_y),
        (doc_x + doc_width, doc_y + fold_size),
        (doc_x + doc_width - fold_size, doc_y + fold_size)
    ], fill=(230, 230, 230, 255))
    
    # Draw lines on document (representing text)
    line_margin = size // 16
    line_height = size // 32
    line_spacing = size // 24
    
    y_pos = doc_y + line_margin * 2
    for i in range(5):
        line_width = doc_width - line_margin * 2
        if i == 0:  # Title line
            line_width = int(line_width * 0.7)
        
        draw.rectangle([
            doc_x + line_margin, 
            y_pos,
            doc_x + line_margin + line_width,
            y_pos + line_height
        ], fill=(100, 100, 100, 255))
        
        y_pos += line_height + line_spacing
    
    # Draw checkmark
    check_size = size // 4
    check_x = doc_x + doc_width - check_size // 2
    check_y = doc_y + doc_height - check_size // 2
    
    # Checkmark circle
    draw.ellipse([
        check_x - check_size//2, 
        check_y - check_size//2,
        check_x + check_size//2,
        check_y + check_size//2
    ], fill=(76, 175, 80, 255))  # Green
    
    # Checkmark
    check_points = [
        (check_x - check_size//4, check_y),
        (check_x - check_size//8, check_y + check_size//8),
        (check_x + check_size//4, check_y - check_size//4)
    ]
    draw.line(check_points, fill=(255, 255, 255, 255), width=size//32)
    
    return img


def create_windows_icon():
    """Create .ico file for Windows"""
    print("Creating Windows icon...")
    
    # Create images at different sizes
    sizes = [16, 32, 48, 64, 128, 256]
    images = []
    
    for size in sizes:
        img = create_icon_image(size)
        images.append(img)
    
    # Save as .ico
    images[0].save('icon.ico', format='ICO', sizes=[(s, s) for s in sizes])
    print("Created icon.ico")


def create_mac_icon():
    """Create .icns file for Mac"""
    print("Creating Mac icon...")
    
    # Create temporary iconset directory
    iconset_dir = Path("DocumentReviewer.iconset")
    iconset_dir.mkdir(exist_ok=True)
    
    # Mac icon sizes (actual size and @2x retina)
    sizes = {
        'icon_16x16.png': 16,
        'icon_16x16@2x.png': 32,
        'icon_32x32.png': 32,
        'icon_32x32@2x.png': 64,
        'icon_128x128.png': 128,
        'icon_128x128@2x.png': 256,
        'icon_256x256.png': 256,
        'icon_256x256@2x.png': 512,
        'icon_512x512.png': 512,
        'icon_512x512@2x.png': 1024,
    }
    
    # Create each size
    for filename, size in sizes.items():
        img = create_icon_image(size)
        img.save(iconset_dir / filename)
    
    # Convert to .icns using iconutil (Mac only)
    import subprocess
    try:
        subprocess.run(['iconutil', '-c', 'icns', str(iconset_dir)], check=True)
        print("Created DocumentReviewer.icns")
        
        # Clean up
        import shutil
        shutil.rmtree(iconset_dir)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Note: iconutil not found. Keeping .iconset folder.")
        print("On Mac, run: iconutil -c icns DocumentReviewer.iconset")


def create_png_logo():
    """Create a PNG logo for general use"""
    print("Creating PNG logo...")
    
    for size in [64, 128, 256, 512]:
        img = create_icon_image(size)
        img.save(f'logo_{size}.png')
    
    print("Created PNG logos")


def main():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("ERROR: Pillow is required to create icons")
        print("Install with: pip install Pillow")
        return
    
    print("Creating application icons...")
    print()
    
    # Create all icon formats
    create_windows_icon()
    
    if os.name == 'posix':  # Mac/Linux
        create_mac_icon()
    
    create_png_logo()
    
    print()
    print("Icons created successfully!")
    print("- icon.ico: Windows application icon")
    if os.name == 'posix':
        print("- DocumentReviewer.icns: Mac application icon")
    print("- logo_*.png: PNG versions for documentation")


if __name__ == "__main__":
    main()