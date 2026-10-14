"""Shared helpers for the s3dlio op-log trace-accuracy harness (stdlib only).

Used by: orchestrate.py (setup/cleanup), cases/*.py (workloads), check.py (verdicts).
Cases run under `python -I`, so they insert H_LIB into sys.path before importing this.
"""
import base64
import hashlib
import json
import os
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Fixtures: objects created NATIVELY (gcloud) before any s3dlio call, each with
# UNIQUE content and a UNIQUE size, so returned bytes / stat size identify the object.
# (name, bucket number, key relative to the run prefix, size).  A key written "/rel"
# is stored under "/<prefix>/rel" (object name with a LEADING SLASH).
# ---------------------------------------------------------------------------
FIXTURES = [
    ("data_a",      1, "data/a.bin",       331),
    ("data_b",      1, "data/b.bin",       347),
    ("data_c",      1, "data/c.bin",       359),
    ("data_d1",     1, "data/dir/d1.bin",  373),
    ("data_d2",     1, "data/dir/d2.bin",  389),
    ("same_b1",     1, "same/obj",         401),
    ("same_b2",     2, "same/obj",         409),
    ("slash_plain", 1, "slash/x",          421),
    ("slash_lead",  1, "/slash/x",         433),   # object name "/<prefix>/slash/x"
    ("dls_plain",   1, "del/ls",           443),
    ("dls_lead",    1, "/del/ls",          449),
    ("del_a_b1",    1, "del/a",            457),   # delete target
    ("del_a_b2",    2, "del/a",            461),   # decoy: same key, other bucket
    ("many_0",      1, "many/m0",          467),
    ("many_1",      1, "many/m1",          479),
    ("many_2",      1, "many/m2",          487),
    ("async_g1",    1, "async/g1",         503),
    ("me_m1",       1, "me/m1",            521),
    ("db_t1",       1, "db/x1",            523),   # delete_batch targets / decoys
    ("db_t2",       2, "db/x2",            541),
    ("db_decoy1",   2, "db/x1",            547),   # decoy in the other bucket, same key as db_t1
    ("db_decoy2",   1, "db/x2",            557),   # decoy in the other bucket, same key as db_t2
]


def make_content(name, size, seed):
    pat = f"<{name}|{seed}>".encode()
    return (pat * (size // len(pat) + 1))[:size]


def fixture_table(prefix, b1, b2, seed):
    """name -> {bucket, key, size, content_b64, sha256}"""
    out = {}
    for name, bn, rel, size in FIXTURES:
        key = ("/" + prefix + rel) if rel.startswith("/") else (prefix + "/" + rel)
        data = make_content(name, size, seed)
        out[name] = dict(
            bucket=b1 if bn == 1 else b2, key=key, size=size,
            content_b64=base64.b64encode(data).decode(),
            sha256=hashlib.sha256(data).hexdigest(),
        )
    return out


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Case context + ground-truth recorder
# ---------------------------------------------------------------------------
def emit_marker(kind, cid):
    # os.write on fd 1: ordered with the Rust tracing output (also written to stdout).
    os.write(1, f"@@H {kind} {cid} {time.time_ns()}\n".encode())


def _jsonable(o):
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(bytes(o)).decode()
    return str(o)


class Ctx:
    def __init__(self):
        e = os.environ
        self.case_dir = e["H_CASE_DIR"]
        self.case = e["H_CASE"]
        self.prefix = e["H_PREFIX"]
        self.b1 = e["H_BUCKET1"]
        self.b2 = e["H_BUCKET2"]
        self.seed = e["H_SEED"]
        self.form = e.get("H_OPLOG_FORM", "path")
        self.call_timeout = float(e.get("H_CALL_TIMEOUT", "25"))
        self.server = e.get("AWS_ENDPOINT_URL", "")
        self.fx = fixture_table(self.prefix, self.b1, self.b2, self.seed)
        self.calls = []
        self.hung = False
        self.init = None
        self.inits = []
        self.final = {}
        self.oplog_path = None
        self.t_proc_start = time.time_ns()

    # --- uris ------------------------------------------------------------
    def uri(self, name):
        f = self.fx[name]
        return f"s3://{f['bucket']}/{f['key']}"

    def content(self, name):
        return base64.b64decode(self.fx[name]["content_b64"])

    def new_key(self, rel, bucket=None, lead=False):
        b = bucket or self.b1
        key = ("/" if lead else "") + self.prefix + "/" + rel
        return b, key

    def asked(self, name, **kw):
        f = self.fx[name]
        d = dict(server=self.server, bucket=f["bucket"], key=f["key"])
        d.update(kw)
        return d

    def asked_new(self, bucket, key, **kw):
        d = dict(server=self.server, bucket=bucket, key=key)
        d.update(kw)
        return d

    # --- op-log ----------------------------------------------------------
    def oplog_init(self, path=None, form=None):
        import s3dlio
        form = form or self.form
        path = path or os.path.join(self.case_dir, "ops.tsv.zst")
        self.oplog_path = path
        arg = path if form == "path" else "file://" + path
        t0 = time.time_ns()
        try:
            s3dlio.init_op_log(arg)
            self.init = dict(arg=arg, form=form, ok=True, active=s3dlio.is_op_log_active(), t_ns=t0)
        except BaseException as ex:  # noqa
            self.init = dict(arg=arg, form=form, ok=False, error=f"{type(ex).__name__}: {str(ex)[:300]}",
                             active=s3dlio.is_op_log_active(), t_ns=t0)
        self.inits.append(self.init)
        self.flush()
        return self.init

    # --- recorder ----------------------------------------------------------
    def call(self, cid, api, uri, fn, *, kind, asked, expect_error=False, args=None,
             expect_bytes=None, note=None, timeout=None, group=None, payload=None, watch=None):
        """Run fn() once, bounded; append the ground-truth entry.

        kind     GET|GET_RANGE|PUT|STAT|LIST|DELETE (what the caller asked for)
        asked    dict: what the CALLER asked s3dlio to do: server ('https://host[:port]' or 'host'),
                 bucket, key (exact; may start with '/'), plus optional 'put_sha256','put_len'
        expect_bytes  exact bytes a correct GET/range must return (None for non-reads)
        group    calls sharing a concurrent window (get_many): same group id
        """
        if payload is not None:
            asked = dict(asked, put_len=len(payload), put_sha256=sha(payload),
                         put_md5_b64=base64.b64encode(hashlib.md5(payload).digest()).decode())
        ent = dict(id=cid, api=api, uri=uri, kind=kind, asked=asked, expect_error=expect_error,
                   args=args or {}, note=note, group=group, pending=True)
        if expect_bytes is not None:
            ent["expect_sha256"] = sha(expect_bytes)
            ent["expect_len"] = len(expect_bytes)
        self.calls.append(ent)
        self.flush()
        box = {}

        def run():
            try:
                box["value"] = fn()
            except BaseException as ex:  # noqa
                box["exc"] = f"{type(ex).__name__}: {str(ex)[:400]}"

        emit_marker("BEGIN", cid)
        ent["t_start_ns"] = time.time_ns()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout or self.call_timeout)
        ent["t_end_ns"] = time.time_ns()
        emit_marker("END", cid)
        if th.is_alive():
            ent["outcome"] = dict(hang=True, ok=False, error=None)
            self.hung = True
        else:
            ent["outcome"] = self._describe(box, ent)
        ent["pending"] = False
        if watch and not th.is_alive():
            ent["native_after"] = self.native_check(watch)
        if asked.get("scheme") == "file" and not th.is_alive():
            pth = asked["key"]
            ent["local_after"] = dict(exists=os.path.exists(pth),
                                      sha256=sha(open(pth, "rb").read()) if os.path.isfile(pth) else None)
        self.flush()
        time.sleep(0.02)  # keep call windows disjoint
        return ent["outcome"]

    @staticmethod
    def native_check(watch):
        """gcloud (Google-native) state of each (bucket, key, role) right after the call.  -> list of dicts"""
        import native
        out = []
        for b, k, role in watch:
            d = native.describe(b, k)
            out.append(dict(bucket=b, key=k, role=role, present=d is not None,
                            md5=(d or {}).get("md5"), size=(d or {}).get("size")))
        return out

    def call_group(self, gid, api, fn, items, *, timeout=None):
        """One concurrent API call (get_many) that stands for several logical calls.
        items: list of dict(id, uri, kind, asked, expect_bytes).  fn() returns [(uri, bytes-like), ...]."""
        ents = []
        for it in items:
            ent = dict(id=it["id"], api=api, uri=it["uri"], kind=it["kind"], asked=it["asked"], expect_error=False,
                       args={}, note=None, group=gid, pending=True)
            if it.get("expect_bytes") is not None:
                ent["expect_sha256"] = sha(it["expect_bytes"])
                ent["expect_len"] = len(it["expect_bytes"])
            ents.append(ent)
            self.calls.append(ent)
        self.flush()
        box = {}

        def run():
            try:
                box["value"] = fn()
            except BaseException as ex:  # noqa
                box["exc"] = f"{type(ex).__name__}: {str(ex)[:400]}"

        emit_marker("BEGIN", gid)
        t0 = time.time_ns()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout or self.call_timeout)
        t1 = time.time_ns()
        emit_marker("END", gid)
        for ent in ents:
            ent["t_start_ns"], ent["t_end_ns"] = t0, t1
            ent["pending"] = False
            if th.is_alive():
                ent["outcome"] = dict(hang=True, ok=False, error=None)
            elif "exc" in box:
                ent["outcome"] = dict(hang=False, ok=False, error=box["exc"])
            else:
                got = dict((u, v) for u, v in box["value"])
                if ent["uri"] in got:
                    b = bytes(got[ent["uri"]])
                    ent["outcome"] = dict(hang=False, ok=True, error=None, len=len(b), sha256=sha(b),
                                          got_b64=base64.b64encode(b[:2048]).decode())
                else:
                    ent["outcome"] = dict(hang=False, ok=False, error="not in result", listing_keys=sorted(got))
        if th.is_alive():
            self.hung = True
        self.flush()
        time.sleep(0.02)

    @staticmethod
    def _describe(box, ent):
        if "exc" in box:
            return dict(hang=False, ok=False, error=box["exc"])
        v = box.get("value")
        out = dict(hang=False, ok=True, error=None)
        k = ent["kind"]
        if k in ("GET", "GET_RANGE"):
            try:
                b = bytes(v)
                out.update(len=len(b), sha256=sha(b), got_b64=base64.b64encode(b[:2048]).decode())
            except Exception as ex:  # noqa
                out.update(len=None, repr=repr(v)[:200])
        elif k == "STAT" and ent["api"] == "exists":
            out["exists"] = bool(v)
        elif k == "STAT":
            out["stat"] = {kk: (vv if isinstance(vv, (int, float, str, bool, type(None))) else str(vv))
                           for kk, vv in dict(v).items()} if isinstance(v, dict) else repr(v)[:300]
        elif k == "LIST":
            out["listing"] = sorted(str(x) for x in v)
        elif k == "DELETE":
            out["repr"] = repr(v)[:200]
        else:
            out["repr"] = repr(v)[:200]
        return out

    def flush(self):
        doc = dict(case=self.case, form=self.form, init=self.init, inits=self.inits, calls=self.calls, hung=self.hung,
                   final=self.final, prefix=self.prefix, b1=self.b1, b2=self.b2,
                   oplog_path=self.oplog_path, t_proc_start_ns=self.t_proc_start)
        tmp = os.path.join(self.case_dir, "ground.json.tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1, default=_jsonable)
        os.replace(tmp, os.path.join(self.case_dir, "ground.json"))

    def finish(self, oplog=True):
        t0 = time.time_ns()
        if not oplog:
            self.final = dict(finalize_ok=None, no_oplog=True, t_ns=t0)
            self.flush()
            sys.stdout.flush()
            if self.hung:
                os._exit(0)
            return
        import s3dlio
        res = {}

        def fin():
            try:
                s3dlio.finalize_op_log()
                res["ok"] = True
            except BaseException as ex:  # noqa
                res["ok"] = False
                res["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"

        th = threading.Thread(target=fin, daemon=True)
        th.start()
        th.join(30)
        self.final = dict(finalize_ok=res.get("ok"), finalize_error=res.get("error"),
                          finalize_hung=th.is_alive(),
                          active_after=(None if th.is_alive() else s3dlio.is_op_log_active()),
                          t_ns=t0)
        self.flush()
        sys.stdout.flush()
        sys.stderr.flush()
        if self.hung or th.is_alive():
            os._exit(0)   # a worker thread is stuck inside a hung call; exit by ourselves, no signals
