#!/usr/bin/env python3
"""Reference CLIP token ids from open_clip's SimpleTokenizer
(pip install open_clip_torch ftfy regex; torch itself is stubbed out)."""
import importlib.util, json, sys, types, os
_t = types.ModuleType("torch"); _t.LongTensor = _t.Tensor = object; sys.modules.setdefault("torch", _t)
path = sys.argv[1]  # .../open_clip/tokenizer.py
spec = importlib.util.spec_from_file_location("octok", path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
tok = mod.SimpleTokenizer(bpe_path=os.path.join(os.path.dirname(path), "bpe_simple_vocab_16e6.txt.gz"))
texts = [
    "a photo of a baby sleeping in a crib",
    "a woman breastfeeding a baby",
    "A Photo of  an EMPTY dark bedroom!!",
    "a mother nursing a baby at her breast",
    "a photo of a baby stretching its arms",
    "parents' 2 kids, 1990s-style ... selfie :)",
]
out = []
for t in texts:
    ids = [tok.sot_token_id] + tok.encode(t) + [tok.eot_token_id]
    ids = ids[:77]; ids[-1] = tok.eot_token_id if len(ids) == 77 else ids[-1]
    ids += [0] * (77 - len(ids))
    out.append([t, ids])
json.dump(out, open(sys.argv[2], "w"))
print("ok", len(out))
