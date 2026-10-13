#!/usr/bin/env python3
"""Regenerates the protocol test vectors in src/tapo/testdata from pytapo
(pip install pytapo==3.4.26), so the Rust port is checked against the
reference implementation."""
import base64, hashlib, hmac, json, os, sys

from pytapo.transport.tpap import spake2p as sp

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "src", "tapo", "testdata")
b64 = lambda b: base64.b64encode(b).decode()

crypt = []
for key, prefix in [
    ("secret", "$5$saltstring"),
    ("Hello world!", "$5$rounds=10000$saltstringsaltstring"),
    ("2e24a3c4f53c4c7b", "$5$abcdefgh$"),
    ("x" * 40, "$5$rounds=1000$roundstoolow"),
]:
    crypt.append((key, prefix, sp.sha256_crypt(key, prefix)))

credential = sp.apply_extra_crypt(sp.md5_hex("cloud-pass"), {"type": "password_shadow", "params": {"passwd_id": 5, "passwd_prefix": "$5$Zm9vYmFyYmF6$"}})
dev_salt = bytes(range(16))
iterations = 1000
dk = hashlib.pbkdf2_hmac("sha256", credential.encode(), dev_salt, iterations, 80)
w0 = int.from_bytes(dk[:40], "big") % sp.N
w1 = int.from_bytes(dk[40:], "big") % sp.N
y = 0x1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF
Y = sp.pt_add(sp.pt_mul(y, sp.G_POINT), sp.pt_mul(w0, sp.N_POINT))
dev_random = bytes(range(100, 132))
user_random = b64(bytes(range(200, 232)))
reg = {"dev_salt": b64(dev_salt), "iterations": iterations, "dev_share": b64(sp.encode_point(Y)), "dev_random": b64(dev_random)}
random = 0xDEADBEEF * (2**200) + 12345
client = sp.Spake2pClient(reg, user_random, credential, random)

# device side of the exchange
X = sp.decode_point(client.Xb)
Z = sp.pt_mul(y, sp.pt_add(X, sp.pt_neg(sp.pt_mul(w0, sp.M_POINT))))
V = sp.pt_mul(y, sp.pt_mul(w1, sp.G_POINT))
context = hashlib.sha256(sp.CONTEXT_TAG + base64.b64decode(user_random) + dev_random).digest()
transcript = sp._len_prefixed(context, b"", b"", sp.encode_point(sp.M_POINT), sp.encode_point(sp.N_POINT),
                              client.Xb, sp.encode_point(Y), sp.encode_point(Z), sp.encode_point(V), w0.to_bytes(32, "big"))
ke = hashlib.sha256(transcript).digest()
conf = sp.hkdf_sha256(ke, None, b"ConfirmationKeys", 64)
dev_confirm = hmac.new(conf[32:], client.Xb, hashlib.sha256).digest()
assert client.device_confirm_ok(dev_confirm)
key, nonce = client.session_key_and_nonce()

json.dump({
    "crypt": crypt,
    "sha1": ["passcode", sp.apply_extra_crypt("passcode", {"type": "password_shadow", "params": {"passwd_id": 2}})],
    "dev_salt": reg["dev_salt"], "iterations": iterations, "dev_share": reg["dev_share"], "dev_random": reg["dev_random"],
    "user_random": user_random, "credential": credential, "random": str(random),
    "xb": b64(client.Xb), "user_confirm": b64(client.user_confirm), "dev_confirm": b64(dev_confirm),
    "key": key.hex(), "nonce": nonce.hex(),
}, open(os.path.join(out, "spake2p.json"), "w"), indent=1)
print("written", out)

# ---- media server (port 8800) digest + AES session keys
from pytapo.media_stream.session import HttpMediaSession
from pytapo.media_stream._utils import pwd_digest
from pytapo.media_stream.crypto import AESHelper

www = 'Digest realm="TP-LINK IP-Camera",nonce="4d2a1b3c5e6f",opaque="0f1e2d3c",encrypt_type="3"'
auth = {i[0].strip().replace('"', ""): i[1].strip().replace('"', "") for i in (j.split("=") for j in www.split(" ", 1)[1].split(","))}
auth.update({"username": "admin", "cnonce": "c0ffee" * 8, "nc": "00000001", "qop": "auth"})
method = HttpMediaSession._get_media_encryption_method(auth)
hashed = pwd_digest(b"cloud-pass", method).decode()
c1 = hashlib.md5(":".join(("admin", auth["realm"], hashed)).encode()).hexdigest()
c2 = hashlib.md5(b"POST:/stream").hexdigest()
response = hashlib.md5(b":".join((c1.encode(), auth["nonce"].encode(), auth["nc"].encode(), auth["cnonce"].encode(), auth["qop"].encode(), c2.encode()))).hexdigest()
kx = 'username="admin" nonce="ABCDEF1234567890"'
aes = AESHelper.from_keyexchange_and_password(kx, "cloud-pass", "", method)
json.dump({"www_authenticate": www, "password": "cloud-pass", "hashed": hashed, "cnonce": auth["cnonce"], "response": response,
           "key_exchange": kx, "key": aes.key.hex(), "iv": aes.iv.hex()}, open(os.path.join(out, "media.json"), "w"), indent=1)
print("media vectors written")
