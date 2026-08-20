import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageFilter, ImageDraw, ImageOps, ImageEnhance

def log(*a):
    print(*a, flush=True)

SRC = "champagne.jpeg"
CUTOUT = "champagne_cutout.png"
OUT = "champagne-pro.jpeg"

log("Loading source image...")
with open(SRC, "rb") as f:
    input_bytes = f.read()

log("Running background removal (plain mask, no matting)...")
session = new_session("u2netp")
result = remove(input_bytes, session=session)
with open(CUTOUT, "wb") as f:
    f.write(result)
log("Background removed, cutout saved.")

bottle = Image.open(CUTOUT).convert("RGBA")

# u2netp occasionally bites a notch out of the silhouette edge (confirmed: the
# raw mask has zero *enclosed* holes, but can have edge-connected bites where
# a low-contrast region -- e.g. the twisted foil against a similarly-lit
# background -- gets misclassified). A bottle's silhouette is convex per
# horizontal row, so fill each row between its own leftmost/rightmost
# foreground pixel to bridge any such notch without touching real edges.
alpha_raw = np.array(bottle.split()[3])
binary = alpha_raw > 30
row_has_fg = binary.any(axis=1)
idx = np.arange(binary.shape[1])
masked_idx = np.where(binary, idx, np.nan)
left = np.nanmin(masked_idx, axis=1)
right = np.nanmax(masked_idx, axis=1)
width = right - left
filled = binary.copy()
for y in np.where(row_has_fg)[0]:
    filled[y, int(left[y]):int(right[y]) + 1] = True

# The notch is an edge *retraction* (the true right edge is simply absent in
# those rows), which row-filling between existing left/right extremes can't
# recover. A bottle photographed near-straight-on is left-right symmetric, so
# estimate the central axis from the widest (most reliable, body/label) rows
# and mirror the mask across it, unioning with the original to recover any
# one-sided bite.
reliable = row_has_fg & (width > np.nanpercentile(width[row_has_fg], 75))
cx = float(np.nanmedian((left[reliable] + right[reliable]) / 2))
log("Estimated symmetry axis at x =", cx)
w = filled.shape[1]
col_idx = np.arange(w)
mirrored_idx = np.clip(np.round(2 * cx - col_idx).astype(int), 0, w - 1)
mirrored = filled[:, mirrored_idx]
symmetrized = filled | mirrored
newly_added = symmetrized & ~binary

rgb_arr = np.array(bottle.convert("RGB"))
mirrored_rgb = rgb_arr[:, mirrored_idx, :]
patched_rgb = np.where(newly_added[..., None], mirrored_rgb, rgb_arr).astype(np.uint8)
patched_alpha = np.where(symmetrized, np.where(binary, alpha_raw, 255), 0).astype(np.uint8)
bottle = Image.merge(
    "RGBA",
    (*Image.fromarray(patched_rgb).split(), Image.fromarray(patched_alpha)),
)
log("Patched", int(newly_added.sum()), "notch pixels via row-fill + mirrored symmetry")

# Feather the patched mask
alpha_np = np.array(bottle.split()[3]).astype(np.float32)
alpha_img = Image.fromarray(alpha_np.astype(np.uint8))
alpha_soft = alpha_img.filter(ImageFilter.GaussianBlur(2))
alpha_soft_np = np.array(alpha_soft).astype(np.float32)
# push mid-tones towards solid/transparent to keep the feather subtle, not muddy
alpha_final = np.clip((alpha_soft_np - 30) * (255.0 / (255 - 30)), 0, 255).astype(np.uint8)
r, g, b, _ = bottle.split()
bottle = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha_final)))

# Tight bbox using a thresholded alpha mask to avoid stray low-alpha halo pixels
mask = alpha_final > 40
ys, xs = np.where(mask)
bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
bottle = bottle.crop(bbox)
log("Cropped to tight bbox:", bottle.size)

# Enhance the bottle: the source photo was dim/low-contrast indoor lighting
rgb = bottle.convert("RGB")
rgb = ImageEnhance.Brightness(rgb).enhance(1.85)
rgb = ImageEnhance.Contrast(rgb).enhance(1.32)
rgb = ImageEnhance.Color(rgb).enhance(1.4)
rgb = ImageEnhance.Sharpness(rgb).enhance(1.2)
bottle = Image.merge("RGBA", (*rgb.split(), bottle.split()[3]))
log("Bottle enhanced.")

# Studio-style relight: warm highlight sweeping the upper-right of the bottle (screen blend)
bw, bh = bottle.size
ryy, rxx = np.mgrid[0:bh, 0:bw]
rcx, rcy = bw * 0.72, bh * 0.18
rmax = max(bw, bh) * 0.65
rd = np.clip(1 - (np.hypot(rxx - rcx, ryy - rcy) / rmax), 0, 1) ** 1.6
highlight = (rd * 150).astype(np.uint8)
highlight_rgb = np.stack([highlight] * 3, axis=-1).astype(np.uint8)
bottle_rgb = np.array(bottle.convert("RGB")).astype(np.float32)
screened = 255 - (255 - bottle_rgb) * (255 - highlight_rgb.astype(np.float32)) / 255
bottle_rgb_final = np.clip(screened, 0, 255).astype(np.uint8)
bottle = Image.merge("RGBA", (*Image.fromarray(bottle_rgb_final).split(), bottle.split()[3]))
log("Bottle relit.")

CANVAS_W, CANVAS_H = 1400, 1800
target_h = int(CANVAS_H * 0.72)
scale = target_h / bottle.height
bottle = bottle.resize((max(1, int(bottle.width * scale)), target_h), Image.LANCZOS)
log("Bottle resized:", bottle.size)

# Elegant premium background: warm gold-brown radial gradient, vectorized with numpy
log("Building background scene...")
yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W]
cx, cy = CANVAS_W * 0.5, CANVAS_H * 0.40
max_r = np.hypot(CANVAS_W, CANVAS_H) * 0.5
d = np.clip(np.hypot(xx - cx, yy - cy) / max_r, 0, 1)

top_color = np.array([168, 126, 68])  # warm gold-brown center (brighter)
mid_color = np.array([82, 58, 32])
edge_color = np.array([30, 20, 14])   # dark but not crushed black

t1 = np.clip(d / 0.55, 0, 1)[..., None]
t2 = np.clip((d - 0.55) / 0.45, 0, 1)[..., None]
near = top_color + (mid_color - top_color) * t1
far = mid_color + (edge_color - mid_color) * t2
bg_arr = np.where(d[..., None] < 0.55, near, far).astype(np.uint8)
bg = Image.fromarray(bg_arr, "RGB")

# Table surface gradient at the bottom for grounding
table = Image.new("L", (CANVAS_W, CANVAS_H), 0)
tdraw = ImageDraw.Draw(table)
table_top = int(CANVAS_H * 0.78)
for y in range(table_top, CANVAS_H):
    alpha = int(150 * (y - table_top) / (CANVAS_H - table_top))
    tdraw.line([(0, y), (CANVAS_W, y)], fill=alpha)
table_layer = Image.new("RGB", (CANVAS_W, CANVAS_H), (5, 3, 2))
bg = Image.composite(table_layer, bg, table)

# Bright warm spotlight glow behind the bottle
glow = Image.new("L", (CANVAS_W, CANVAS_H), 0)
gdraw = ImageDraw.Draw(glow)
gw, gh = 900, 1150
gx = CANVAS_W // 2 - gw // 2
gy = int(CANVAS_H * 0.04)
gdraw.ellipse([gx, gy, gx + gw, gy + gh], fill=235)
glow = glow.filter(ImageFilter.GaussianBlur(140))
gold_layer = Image.new("RGB", (CANVAS_W, CANVAS_H), (232, 184, 96))
bg = Image.composite(gold_layer, bg, glow)

# Secondary tighter, brighter highlight right behind the bottle for a rim-lit look
glow2 = Image.new("L", (CANVAS_W, CANVAS_H), 0)
g2draw = ImageDraw.Draw(glow2)
gw2, gh2 = 420, 1300
gx2 = CANVAS_W // 2 - gw2 // 2
gy2 = int(CANVAS_H * 0.05)
g2draw.ellipse([gx2, gy2, gx2 + gw2, gy2 + gh2], fill=140)
glow2 = glow2.filter(ImageFilter.GaussianBlur(90))
gold_layer2 = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 210, 130))
bg = Image.composite(gold_layer2, bg, glow2)

bg = bg.filter(ImageFilter.GaussianBlur(1))
log("Background scene ready.")

bg = bg.convert("RGBA")

bottle_pos_x = (CANVAS_W - bottle.width) // 2
bottle_pos_y = CANVAS_H - target_h - int(CANVAS_H * 0.05)

# Drop shadow for bottle
shadow_alpha = bottle.split()[3].filter(ImageFilter.GaussianBlur(20))
shadow = Image.new("RGBA", bottle.size, (0, 0, 0, 0))
shadow.putalpha(shadow_alpha)
bg.alpha_composite(shadow, (bottle_pos_x + 20, bottle_pos_y + 28))

# Contact shadow on the table
contact = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
cdraw = ImageDraw.Draw(contact)
ellipse_cy = bottle_pos_y + bottle.height - 8
ew, eh = int(bottle.width * 0.95), 55
cdraw.ellipse([CANVAS_W // 2 - ew // 2, ellipse_cy - eh // 2, CANVAS_W // 2 + ew // 2, ellipse_cy + eh // 2], fill=(0, 0, 0, 160))
contact = contact.filter(ImageFilter.GaussianBlur(22))
bg.alpha_composite(contact)

# Soft reflection of the bottle on the "table" (classic premium product-shot touch)
reflection = ImageOps.flip(bottle)
refl_alpha = np.array(reflection.split()[3]).astype(np.float32)
h = refl_alpha.shape[0]
fade = np.linspace(0.38, 0.0, h)[:, None]
refl_alpha = (refl_alpha * fade).astype(np.uint8)
reflection = Image.merge("RGBA", (*reflection.split()[:3], Image.fromarray(refl_alpha)))
reflection = reflection.filter(ImageFilter.GaussianBlur(3))
bg.alpha_composite(reflection, (bottle_pos_x, bottle_pos_y + bottle.height))

# Paste bottle
bg.alpha_composite(bottle, (bottle_pos_x, bottle_pos_y))

# Subtle vignette: bright "protected" zone at center (255), blurred, then
# inverted so alpha (= how much black to lay down) is near 0 at center and
# only rises towards the far corners. Bounds must fall INSIDE the canvas or
# the falloff never happens within frame (previous bug: bounds were far
# larger than the canvas, so the mask was ~uniform and, after inversion,
# washed the whole image with near-opaque black instead of just the edges).
vign_mask = Image.new("L", (CANVAS_W, CANVAS_H), 0)
vdraw2 = ImageDraw.Draw(vign_mask)
vdraw2.ellipse([-CANVAS_W * 0.15, -CANVAS_H * 0.1, CANVAS_W * 1.15, CANVAS_H * 0.95], fill=255)
vign_mask = vign_mask.filter(ImageFilter.GaussianBlur(220))
vign_mask = ImageOps.invert(vign_mask)
black_layer = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 255))
bg.alpha_composite(Image.merge("RGBA", (*black_layer.split()[:3], vign_mask)))

final = bg.convert("RGB")
final.save(OUT, quality=93)
log("done", final.size)
