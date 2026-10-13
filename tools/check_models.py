#!/usr/bin/env python3
"""Reference implementation of the frame analysis (nudenet.py preprocessing,
CLIP/PIL preprocessing) to cross-check `tapdream inspect`.

usage: check_models.py IMAGE NUDENET.onnx CLIP_VISUAL.onnx CLIP_TEXTUAL.onnx OPEN_CLIP_TOKENIZER.py
The image is first letterboxed to 1920x1080 yuv420p by ffmpeg exactly like
tapdream does, then converted to RGB by ffmpeg."""
import importlib.util, subprocess, sys, types, os
import numpy as np, cv2, onnxruntime as ort
from PIL import Image

img, nn_path, vis_path, txt_path, tok_path = sys.argv[1:6]
W, H = 1920, 1080
vf = f"fps=1:eof_action=pass,scale={W}:{H}:force_original_aspect_ratio=decrease:flags=area:out_color_matrix=bt709:out_range=tv,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
yuv = subprocess.run(["ffmpeg", "-v", "error", "-i", img, "-vf", vf, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], capture_output=True, check=True).stdout
rgb = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{W}x{H}", "-color_range", "tv", "-colorspace", "bt709",
                      "-i", "-", "-vf", "scale=in_color_matrix=bt709:in_range=tv,format=rgb24", "-f", "rawvideo", "-"], input=yuv, capture_output=True, check=True).stdout
rgb = np.frombuffer(rgb, np.uint8).reshape(H, W, 3)

# --- NudeNet exactly as nudenet.py (_read_image + blobFromImage)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
side = max(bgr.shape[:2])
pad = cv2.copyMakeBorder(bgr, 0, side - H, 0, side - W, cv2.BORDER_CONSTANT)
blob = cv2.dnn.blobFromImage(pad, 1 / 255.0, (320, 320), (0, 0, 0), swapRB=True, crop=False)
out = ort.InferenceSession(nn_path).run(None, {"images": blob})[0][0]
labels = ["FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
          "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
          "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED"]
for i, l in enumerate(labels):
    m = out[4 + i].max()
    if m >= 0.05:
        print(f"  {l:<26} {m:.3f}")

# --- CLIP as OpenAI preprocess (bicubic resize of the short side, center crop)
im = Image.fromarray(rgb)
s = 224 / min(im.size)
im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
l, t = (im.size[0] - 224) // 2, (im.size[1] - 224) // 2
x = np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255.0
x = (x - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
x = x.transpose(2, 0, 1)[None].astype(np.float32)
emb = ort.InferenceSession(vis_path).run(None, {"pixel_values": x})[0][0]
emb /= np.linalg.norm(emb)
np.save("/tmp/ref_clip_emb.npy", emb)
print("clip embedding[:4]", np.round(emb[:4], 4))
