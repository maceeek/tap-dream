#!/usr/bin/env python3
"""A fake Tapo camera for testing tapdream without hardware.

Implements the parts of the protocol tapdream uses:
  * control API over HTTPS: classic encrypt_type 3 login + securePassthrough
    (--transport classic) or TPAP SPAKE2+ + AES-CCM (--transport tpap)
  * media server: digest auth, key exchange, AES-CBC encrypted multipart
    stream of MPEG-TS for the `download` request

Recordings are MPEG-TS files named <start>-<end>.ts (unix seconds) in --recordings.

    python3 tools/mock_camera.py --recordings DIR --password secret

Needs pycryptodome; TPAP mode additionally needs pytapo (for the EC math).
"""
import argparse, base64, hashlib, hmac, json, os, random, socket, ssl, struct, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

ap = argparse.ArgumentParser()
ap.add_argument("--recordings", required=True)
ap.add_argument("--password", default="secret")
ap.add_argument("--control-port", type=int, default=4443)
ap.add_argument("--media-port", type=int, default=18800)
ap.add_argument("--transport", choices=["classic", "tpap"], default="classic")
ap.add_argument("--clock-offset", type=int, default=0, help="camera clock = real clock - offset")
ap.add_argument("--chunk", type=int, default=188 * 200)
ap.add_argument("--old-firmware", action="store_true", help="refuse the fast download request (only playback)")
args = ap.parse_args()

PW = args.password
H_MD5 = hashlib.md5(PW.encode()).hexdigest().upper()
H_SHA = hashlib.sha256(PW.encode()).hexdigest().upper()
STOK = "mockstok" + os.urandom(4).hex()
USER_ID = 4242
state = {"cnonce": None, "nonce": None, "lsk": None, "ivb": None, "seq": None, "tpap": None}
log = lambda *a: print("[mock]", *a, file=sys.stderr, flush=True)


def recordings():
    out = []
    for f in sorted(os.listdir(args.recordings)):
        if f.endswith(".ts"):
            s, e = f[:-3].split("-")
            out.append((int(s) - args.clock_offset, int(e) - args.clock_offset, os.path.join(args.recordings, f)))
    return out


def handle_method(m, params):
    if m == "getDeviceInfo":
        return {"device_info": {"basic_info": {"device_type": "SMART.IPCAMERA", "device_model": "C200-MOCK", "sw_version": "1.0.0"}}}
    if m == "getClockStatus":
        return {"system": {"clock_status": {"seconds_from_1970": int(time.time()) - args.clock_offset}}}
    if m == "getUserID":
        return {"user_id": USER_ID}
    if m == "searchDateWithVideo":
        days = sorted({time.strftime("%Y%m%d", time.localtime(s)) for s, _, _ in recordings()})
        return {"playback": {"search_results": [{f"search_results_{i}": {"date": d}} for i, d in enumerate(days)]}}
    if m == "searchVideoOfDay":
        q = params["playback"]["search_video_utility"]
        assert q["id"] == USER_ID, q
        res = [{f"search_video_results_{i}": {"startTime": s, "endTime": e, "vedio_type": 2}}
               for i, (s, e, _) in enumerate(r for r in recordings() if time.strftime("%Y%m%d", time.localtime(r[0])) == q["date"])]
        return {"playback": {"search_video_results": res}}
    raise KeyError(m)


def run_requests(req):
    assert req["method"] == "multipleRequest", req
    responses = []
    for r in req["params"]["requests"]:
        try:
            responses.append({"method": r["method"], "result": handle_method(r["method"], r.get("params")), "error_code": 0})
        except KeyError:
            responses.append({"method": r["method"], "error_code": -40210})
    return {"result": {"responses": responses}, "error_code": 0}


def token(kind, cnonce, nonce):
    hk = hashlib.sha256((cnonce + H_SHA + nonce).encode()).hexdigest().upper()
    return hashlib.sha256((kind + cnonce + nonce + hk).encode()).digest()[:16]


# ---------------------------------------------------------------- TPAP (device side)
if args.transport == "tpap":
    from pytapo.transport.tpap import spake2p as sp


def tpap_register(params):
    st = {}
    st["user_random"] = base64.b64decode(params["user_random"])
    salt, iters = os.urandom(16), 1000
    prefix = "$5$" + base64.b64encode(os.urandom(9)).decode().replace("+", ".").replace("/", ".") + "$"
    cred = sp.apply_extra_crypt(sp.md5_hex(PW), {"type": "password_shadow", "params": {"passwd_id": 5, "passwd_prefix": prefix}})
    dk = hashlib.pbkdf2_hmac("sha256", cred.encode(), salt, iters, 80)
    st["w0"] = int.from_bytes(dk[:40], "big") % sp.N
    st["w1"] = int.from_bytes(dk[40:], "big") % sp.N
    st["y"] = random.randrange(1, sp.N)
    st["Y"] = sp.pt_add(sp.pt_mul(st["y"], sp.G_POINT), sp.pt_mul(st["w0"], sp.N_POINT))
    st["dev_random"] = os.urandom(32)
    state["tpap"] = st
    return {"dev_salt": base64.b64encode(salt).decode(), "iterations": iters, "dev_share": base64.b64encode(sp.encode_point(st["Y"])).decode(),
            "dev_random": base64.b64encode(st["dev_random"]).decode(), "cipher_suites": 1, "encryption": "aes_128_ccm",
            "extra_crypt": {"type": "password_shadow", "params": {"passwd_id": 5, "passwd_prefix": prefix}}}


def tpap_share(params):
    st = state["tpap"]
    xb = base64.b64decode(params["user_share"])
    X = sp.decode_point(xb)
    Z = sp.pt_mul(st["y"], sp.pt_add(X, sp.pt_neg(sp.pt_mul(st["w0"], sp.M_POINT))))
    V = sp.pt_mul(st["y"], sp.pt_mul(st["w1"], sp.G_POINT))
    context = hashlib.sha256(sp.CONTEXT_TAG + st["user_random"] + st["dev_random"]).digest()
    tr = sp._len_prefixed(context, b"", b"", sp.encode_point(sp.M_POINT), sp.encode_point(sp.N_POINT), xb, sp.encode_point(st["Y"]),
                          sp.encode_point(Z), sp.encode_point(V), st["w0"].to_bytes(32, "big"))
    ke = hashlib.sha256(tr).digest()
    conf = sp.hkdf_sha256(ke, None, b"ConfirmationKeys", 64)
    if not hmac.compare_digest(base64.b64decode(params["user_confirm"]), hmac.new(conf[:32], sp.encode_point(st["Y"]), hashlib.sha256).digest()):
        return None
    shared = sp.hkdf_sha256(ke, None, b"SharedKey", 32)
    st["key"] = sp.hkdf_sha256(shared, b"tp-kdf-salt-aes128-key", b"tp-kdf-info-aes128-key", 32)[:16]
    st["nonce"] = sp.hkdf_sha256(shared, b"tp-kdf-salt-aes128-iv", b"tp-kdf-info-aes128-iv", 32)[:12]
    st["seq"] = 77
    return {"dev_confirm": base64.b64encode(hmac.new(conf[32:], xb, hashlib.sha256).digest()).decode(), "stok": STOK, "start_seq": 77, "expired": 3600}


class Control(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def reply(self, obj, status=200, raw=None, ctype="application/json"):
        body = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/":
            req = json.loads(body)
            p = req.get("params", {})
            if args.transport == "tpap":
                sub = p.get("sub_method")
                if sub == "discover":
                    return self.reply({"error_code": 0, "result": {"tpap": {"pake": [2], "user_hash_type": 0}}})
                if sub == "pake_register":
                    assert p["username"] == hashlib.md5(b"admin").hexdigest(), p
                    return self.reply({"error_code": 0, "result": tpap_register(p)})
                if sub == "pake_share":
                    r = tpap_share(p)
                    if r is None:
                        log("TPAP: wrong password")
                        return self.reply({"error_code": -40401})
                    log("TPAP login ok")
                    return self.reply({"error_code": 0, "result": r})
                return self.reply({"error_code": -40211})
            if "digest_passwd" in p:
                expect = hashlib.sha256((H_SHA + state["cnonce"] + state["nonce"]).encode()).hexdigest().upper() + state["cnonce"] + state["nonce"]
                if p["digest_passwd"] != expect:
                    return self.reply({"error_code": -40411, "result": {"data": {"code": -40411}}}, 401)
                state["lsk"], state["ivb"] = token("lsk", state["cnonce"], state["nonce"]), token("ivb", state["cnonce"], state["nonce"])
                state["seq"] = 100
                log("classic login ok")
                return self.reply({"error_code": 0, "result": {"stok": STOK, "user_group": "root", "start_seq": 100}})
            if p.get("encrypt_type") == "3":
                state["cnonce"], state["nonce"] = p["cnonce"], os.urandom(8).hex().upper()
                confirm = hashlib.sha256((p["cnonce"] + H_SHA + state["nonce"]).encode()).hexdigest().upper() + state["nonce"] + p["cnonce"]
                return self.reply({"error_code": -40413, "result": {"data": {"code": -40413, "encrypt_type": ["3"], "key": "x", "nonce": state["nonce"], "device_confirm": confirm}}})
            return self.reply({"error_code": -40210})
        if self.path != f"/stok={STOK}/ds":
            return self.reply({"error_code": -40401})
        if args.transport == "tpap":
            st = state["tpap"]
            seq = struct.unpack(">I", body[:4])[0]
            c = AES.new(st["key"], AES.MODE_CCM, nonce=st["nonce"][:8] + struct.pack(">I", seq), mac_len=16)
            req = json.loads(c.decrypt_and_verify(body[4:-16], body[-16:]))
            out = json.dumps(run_requests(req)).encode()
            rs = seq + 1000
            c = AES.new(st["key"], AES.MODE_CCM, nonce=st["nonce"][:8] + struct.pack(">I", rs), mac_len=16)
            return self.reply(None, raw=struct.pack(">I", rs) + c.encrypt(out) + c.digest(), ctype="application/octet-stream")
        outer = json.loads(body)
        seq = int(self.headers["Seq"])
        assert seq == state["seq"], (seq, state["seq"])
        tag = hashlib.sha256((H_SHA + state["cnonce"]).encode()).hexdigest().upper()
        tag = hashlib.sha256(tag.encode() + body + str(seq).encode()).hexdigest().upper()
        if self.headers["Tapo_tag"] != tag:
            log("bad Tapo_tag")
            return self.reply({"error_code": -40401})
        state["seq"] += 1
        req = json.loads(unpad(AES.new(state["lsk"], AES.MODE_CBC, state["ivb"]).decrypt(base64.b64decode(outer["params"]["request"])), 16))
        out = json.dumps(run_requests(req)).encode()
        enc = AES.new(state["lsk"], AES.MODE_CBC, state["ivb"]).encrypt(pad(out, 16))
        return self.reply({"error_code": 0, "result": {"response": base64.b64encode(enc).decode()}})


# ---------------------------------------------------------------- media server
def media_conn(sock):
    f = sock.makefile("rb")

    def read_head():
        lines = []
        while True:
            l = f.readline()
            if not l:
                raise EOFError
            if l in (b"\r\n", b"\n"):
                return lines
            lines.append(l.decode().strip())

    head = read_head()
    nonce, realm, opaque = os.urandom(8).hex(), "TP-LINK IP-Camera", os.urandom(4).hex()
    sock.sendall(f'HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Digest realm="{realm}",nonce="{nonce}",opaque="{opaque}",encrypt_type="3"\r\n\r\n'.encode())
    head = read_head()
    auth = next(l for l in head if l.startswith("Authorization:")).split(" ", 2)[2]
    a = {k.strip(): v.strip().strip('"') for k, v in (x.split("=", 1) for x in auth.split(","))}
    c1 = hashlib.md5(f"admin:{realm}:{H_SHA}".encode()).hexdigest()
    c2 = hashlib.md5(b"POST:/stream").hexdigest()
    expect = hashlib.md5(f"{c1}:{nonce}:{a['nc']}:{a['cnonce']}:{a['qop']}:{c2}".encode()).hexdigest()
    if a["response"] != expect or a["opaque"] != opaque:
        sock.sendall(b"HTTP/1.1 401 Unauthorized\r\n\r\n")
        return
    kx = os.urandom(8).hex()
    key = hashlib.md5(f"{kx}:{H_SHA}".encode()).digest()
    iv = hashlib.md5(f"admin:{kx}".encode()).digest()
    sock.sendall(f'HTTP/1.1 200 OK\r\nContent-Type: multipart/mixed;boundary=--device-stream-boundary--\r\nKey-Exchange: username="admin" nonce="{kx}"\r\n\r\n'.encode())
    # first client part: the request
    boundary = f.readline()
    assert b"client-stream-boundary" in boundary, boundary
    h = {k: v.strip() for k, v in (l.split(":", 1) for l in read_head())}
    req = json.loads(f.read(int(h["Content-Length"])))
    f.readline()
    window = int(h.get("X-Data-Window-Size", "200"))
    acks = []

    def drain():
        try:
            while True:
                l = f.readline()
                if not l:
                    return
                if b"X-Data-Received" in l:
                    acks.append(int(l.split(b":")[1]))
        except Exception:
            return

    threading.Thread(target=drain, daemon=True).start()

    def part(mime, payload, extra=""):
        sock.sendall(f"--device-stream-boundary--\r\nContent-Type: {mime}\r\nContent-Length: {len(payload)}\r\n{extra}\r\n".encode() + payload + b"\r\n")

    dl = req["params"].get("download")
    playback = req["params"].get("playback")
    if dl is None and playback is None or (dl is not None and args.old_firmware):
        part("application/json", json.dumps({"type": "response", "seq": req["seq"], "params": {"error_code": -40106}}).encode())
        time.sleep(0.5)
        return
    dl = dl or playback
    assert dl["client_id"] == USER_ID, dl
    start, end = int(dl["start_time"]), int(dl["end_time"])
    match = [p for s, e, p in recordings() if s == start and e == end]
    part("application/json", json.dumps({"type": "response", "seq": req["seq"], "params": {"error_code": 0, "session_id": "321"}}).encode())
    data = open(match[0], "rb").read() if match else b""
    log(f"streaming {len(data)} bytes for {start}-{end}, window {window}")
    seq = 0
    for i in range(0, len(data), args.chunk):
        seq += 1
        enc = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(data[i:i + args.chunk], 16))
        part("video/mp2t", enc, f"X-If-Encrypt: 1\r\nX-Session-Id: 321\r\nX-Data-Sequence: {seq}\r\n")
    if playback is not None:
        time.sleep(30)  # playback streams do not end on their own
        return
    part("application/json", json.dumps({"type": "notification", "params": {"event_type": "stream_status", "status": "finished"}}).encode())
    time.sleep(0.2)
    log(f"done, {seq} parts, acks received: {len(acks)}")


def media_server():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.media_port))
    srv.listen(4)
    while True:
        c, _ = srv.accept()

        def run(c=c):
            try:
                media_conn(c)
            except Exception as e:
                log("media error:", repr(e))
            finally:
                try:
                    c.close()
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()


def main():
    d = tempfile.mkdtemp()
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=tapo", "-days", "2",
                    "-keyout", f"{d}/k.pem", "-out", f"{d}/c.pem"], check=True, capture_output=True)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(f"{d}/c.pem", f"{d}/k.pem")
    httpd = ThreadingHTTPServer(("127.0.0.1", args.control_port), Control)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=media_server, daemon=True).start()
    log(f"control https://127.0.0.1:{args.control_port} ({args.transport}), media 127.0.0.1:{args.media_port}, {len(recordings())} recordings")
    httpd.serve_forever()


main()
