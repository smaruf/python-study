# App Icons

MicroPythonOS displays a **48 × 48 pixel PNG** icon for each app in the launcher.

Place your icon file here as `icon-48.png`.

---

## Generating an icon

### Option A — Use any image editor

1. Create a 48 × 48 px canvas.
2. Design your icon (transparent background is supported).
3. Export as `icon-48.png` and save it in this `res/` folder.

### Option B — Convert an existing image with Pillow (Python)

```bash
pip install Pillow

python3 - <<'EOF'
from PIL import Image

src = "my_source_image.png"   # path to your source image
dst = "res/icon-48.png"       # output path

img = Image.open(src).convert("RGBA")
img = img.resize((48, 48), Image.LANCZOS)
img.save(dst)
print(f"Saved {dst}")
EOF
```

### Option C — Use ImageMagick (command line)

```bash
convert my_source_image.png -resize 48x48 res/icon-48.png
```

---

## Icon requirements

| Property | Value |
|----------|-------|
| Format | PNG |
| Dimensions | 48 × 48 px |
| Colour mode | RGBA (transparency supported) or RGB |
| File name | `icon-48.png` (exact name expected by the OS) |

---

## Placeholder

Until you add a real icon, MicroPythonOS will show a default placeholder icon
for apps that have no `res/icon-48.png`.  The app will still launch correctly.
