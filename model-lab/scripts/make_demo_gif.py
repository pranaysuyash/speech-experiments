#!/usr/bin/env python3
"""
Generate a small terminal-style demo GIF for README showing starting backend + frontend.
"""
from PIL import Image, ImageDraw, ImageFont
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'demo.gif')
OUT = os.path.abspath(OUT)

# Simple terminal font fallback
try:
    font = ImageFont.truetype("Menlo.ttc", 14)
except Exception:
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

frames = []
width, height = 640, 200
bg = (8, 10, 12)
accent = (6, 182, 212)
text_color = (230, 237, 243)
small = 12

scenes = [
    [
        ("$ chmod +x scripts/start_demo.sh",),
        ("$ ./scripts/start_demo.sh",),
    ],
    [
        ("Starting backend -> http://127.0.0.1:8000",),
        ("health: {\"status\": \"ok\"}",),
    ],
    [
        ("Starting frontend -> http://localhost:5173",),
        ("Open browser to http://localhost:5173/",),
    ],
    [
        ("Done. Ctrl-C to stop. Follow for updates.",),
    ],
]

for scene in scenes:
    # for each scene make 6 frames for a simple fade/typing feel
    for step in range(6):
        im = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(im)
        # header
        draw.rectangle([0, 0, width, 36], fill=(16, 18, 23))
        draw.text((12, 10), "Model Lab — Demo", font=font, fill=accent)
        # draw lines
        y = 52
        for idx, line in enumerate(scene):
            # simulate typing by drawing partial text according to step
            text = line[0]
            cut = int(len(text) * (step / 5.0))
            draw.text((18, y + idx * 22), text[:cut], font=font, fill=text_color)
        frames.append(im)

# Save GIF
frames[0].save(OUT, save_all=True, append_images=frames[1:], optimize=True, duration=200, loop=0)
print("Wrote", OUT)
