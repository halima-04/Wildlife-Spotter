"""Image perturbations that SIMULATE hard field conditions: night, infrared camera traps, distant
animals, dusty haze and low golden light.

Pure PIL + numpy (no torch, no scikit-learn), so the dashboard can import it on any laptop.

IMPORTANT (say this in your report): these are simulations. They are useful for testing and for
training robustness, but they are not a substitute for real night / camera-trap photos.
"""
import numpy as np
from PIL import Image, ImageFilter


def _arr(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


# ----------------------------------------------------------------------------- conditions
def night(img, rng, level=0.7):
    """Low light with a cold blue cast, less colour and sensor noise."""
    a = _arr(img) ** (2.0 + 1.5 * level)
    a = a * np.array([0.40, 0.50, 0.70], dtype=np.float32)
    grey = a.mean(axis=2, keepdims=True)
    a = grey + (a - grey) * 0.5
    a = a + rng.normal(0, 0.008 + 0.025 * level, a.shape).astype(np.float32)
    return _img(a)


def infrared(img, rng, level=0.7):
    """Camera-trap flash at night: greyscale, bright centre, dark edges, grain."""
    a = _arr(img)
    g = 0.30 * a[..., 0] + 0.59 * a[..., 1] + 0.11 * a[..., 2]
    g = np.clip((g - 0.5) * (1.2 + 0.5 * level) + 0.5, 0, 1)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    g = g * np.clip(1.1 - 0.45 * level * r2, 0.2, 1.0)
    g = g + rng.normal(0, 0.02 + 0.03 * level, g.shape).astype(np.float32)
    return _img(np.stack([g, g, g], axis=2))


def far(img, rng, scale=0.35):
    """Animal far away: the photo is shrunk into the frame and the surroundings fill the rest.

    The surroundings are built from the row-wise median colour of the photo's left and right edges (sky / horizon / ground bands),
    so no blurred 'ghost' of the big animal is left behind. The shrunken photo is blended in softly.
    """
    w, h = img.size
    scale = float(np.clip(scale, 0.1, 1.0))
    a = _arr(img)
    edges = np.concatenate([a[:, : max(1, w // 4)], a[:, 3 * w // 4:]], axis=1)   # left/right edges, animal is usually centred
    rows = np.median(edges, axis=1, keepdims=True)                 # (h, 1, 3)
    bg_a = np.repeat(rows, w, axis=1) + rng.normal(0, 0.03, a.shape).astype(np.float32)
    bg = _img(bg_a).filter(ImageFilter.GaussianBlur(radius=max(1.0, h / 60)))
    sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
    small = img.resize((sw, sh), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
    x = int((w - sw) * rng.uniform(0.2, 0.8))
    y = int((h - sh) * rng.uniform(0.3, 0.8))
    canvas = bg.copy()
    canvas.paste(small, (x, y))
    feather = max(1.0, sw * 0.08)
    inset = int(feather * 1.5)
    mask = Image.new("L", (w, h), 0)
    mask.paste(255, (x + inset, y + inset, max(x + inset + 1, x + sw - inset), max(y + inset + 1, y + sh - inset)))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))   # soft edge: no hard rectangle
    return Image.composite(canvas, bg, mask)


def haze(img, rng, level=0.7):
    """Dust or heat haze: washed-out colours, low contrast, slight blur."""
    a = _arr(img)
    alpha = 0.15 + 0.40 * level
    a = a * (1 - alpha) + np.array([0.90, 0.82, 0.70], dtype=np.float32) * alpha
    a = (a - 0.5) * (1 - 0.3 * level) + 0.5
    return _img(a).filter(ImageFilter.GaussianBlur(0.5 + level))


def golden_hour(img, rng, level=0.7):
    """Low warm sun at dawn or dusk: orange cast, darker, higher contrast."""
    a = _arr(img) * np.array([1.0 + 0.2 * level, 0.95, 1.0 - 0.35 * level], dtype=np.float32)
    a = ((a - 0.5) * (1.0 + 0.2 * level) + 0.5) * (1.0 - 0.15 * level)
    return _img(a)


# fixed settings used when TESTING models, so every model is tested on identical conditions
CONDITIONS = {
    "clean": lambda im, r: im,
    "night": lambda im, r: night(im, r, 0.7),
    "infrared": lambda im, r: infrared(im, r, 0.7),
    "far": lambda im, r: far(im, r, 0.40),
    "very_far": lambda im, r: far(im, r, 0.22),
    "far_night": lambda im, r: night(far(im, r, 0.35), r, 0.7),
    "haze": lambda im, r: haze(im, r, 0.7),
    "golden_hour": lambda im, r: golden_hour(im, r, 0.7),
}
LABELS = {"clean": "Clean", "night": "Night", "infrared": "Infrared trap", "far": "Far (40%)",
          "very_far": "Very far (22%)", "far_night": "Far + night", "haze": "Dust haze",
          "golden_hour": "Golden hour"}


def random_condition(img, rng):
    """Random condition with random strength, used while TRAINING the robust model."""
    name = rng.choice(["night", "infrared", "far", "far_night", "haze", "golden_hour"])
    lvl = rng.uniform(0.3, 1.0)
    if name == "night":
        return night(img, rng, lvl), name
    if name == "infrared":
        return infrared(img, rng, lvl), name
    if name == "far":
        return far(img, rng, rng.uniform(0.18, 0.6)), name
    if name == "far_night":
        return night(far(img, rng, rng.uniform(0.18, 0.6)), rng, lvl), name
    if name == "haze":
        return haze(img, rng, lvl), name
    return golden_hour(img, rng, lvl), name


TIMES_OF_DAY = ["Day", "Golden hour", "Night", "Infrared camera trap"]


def stress(img, time_of_day="Day", scale=1.0, haze_level=0.0, seed=0):
    """Combine conditions for the dashboard sliders. scale=1.0 means the animal fills the frame."""
    rng = np.random.default_rng(seed)
    out = img.convert("RGB")
    if scale < 0.98:
        out = far(out, rng, scale)
    if haze_level > 0.01:
        out = haze(out, rng, min(1.0, haze_level / 0.7))
    if time_of_day == "Golden hour":
        out = golden_hour(out, rng, 0.7)
    elif time_of_day == "Night":
        out = night(out, rng, 0.7)
    elif time_of_day == "Infrared camera trap":
        out = infrared(out, rng, 0.7)
    return out
