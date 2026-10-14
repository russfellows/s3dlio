#!/usr/bin/env python3
"""Trace-accuracy checker for s3dlio's op-log.  Stdlib only; run through `uv run ... python -I`.

For every call the program made it reports, SEPARATELY:
  (1) ACTION  - did s3dlio DO what the caller asked?  (server / scheme / bucket / key / object)
  (2) RECORD  - does the op-log say what actually happened?  (exactly one row; right fields)
Truth sources, in this order:  the program's own ground-truth call list; Google-native state
(gcloud snapshots before/after, unique object contents/sizes); s3dlio's own client debug log.
Never a local test server.  Consumer tools (sai3-bench, polarWarp, warp-replay) are not used.

usage: check.py RUN_DIR [--known known.json] [--quiet]
"""
import argparse
import base64
import calendar
import fnmatch
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
CRED = re.compile(r"^(GOOG1|AKIA|ASIA)[A-Z0-9]{8,}$")

# --------------------------------------------------------------------------- client debug log
RE_MARK = re.compile(r"^@@H (BEGIN|END) (\S+) (\d+)$")
RE_ATT = re.compile(r"S3\.(\w+)\{[^}]*sdk_invocation_id=(\d+)[^}]*\}:try_op: beginning attempt #(\d+)")
RE_EP = re.compile(r'S3\.(\w+)\{[^}]*sdk_invocation_id=(\d+)[^}]*\}.*will apply endpoint Endpoint \{ url: "([^"]+)"')
RE_STAT = re.compile(r"stat_object: bucket='([^']*)', key='([^']*)'")
RE_S3L = re.compile(r"S3 (GET|PUT) s3://([^/]+)/(.*) \((\d+) bytes\)")
RE_OPURI = re.compile(r"S3ObjectStore::(\w+) (?:uri|prefix)='([^']*)'")
RE_LISTC = re.compile(r"list_objects_stream called: bucket='([^']*)', path='([^']*)'")
RE_CLIENT = re.compile(r"(Per-endpoint )?S3 client ready \(path-style: forced, endpoint: ([^)]+)\)")


def parse_client_log(path):
    segs, cur, total_att = {}, None, 0
    lines = []
    if os.path.exists(path):
        raw = open(path, "rb").read().decode("utf-8", "replace")
        lines = [ANSI.sub("", ln) for ln in raw.splitlines()]
    seen = set()
    for ln in lines:
        m = RE_MARK.match(ln)
        if m:
            kind, cid = m.group(1), m.group(2)
            if kind == "BEGIN":
                cur = cid
                segs.setdefault(cid, [])
            else:
                cur = None
            continue
        a = RE_ATT.search(ln)
        if a:
            total_att += 1
        if cur is not None:
            segs[cur].append(ln)
    return segs, total_att, lines


def seg_evidence(lines):
    ev = dict(sdk=[], stat=[], s3=[], opuri=[], listc=[], clients=[])
    inv = {}
    for ln in lines:
        m = RE_ATT.search(ln)
        if m:
            k = (m.group(1), m.group(2))
            d = inv.setdefault(k, dict(op=m.group(1), inv=m.group(2), url=None, attempts=0))
            d["attempts"] += 1
        m = RE_EP.search(ln)
        if m:
            k = (m.group(1), m.group(2))
            d = inv.setdefault(k, dict(op=m.group(1), inv=m.group(2), url=None, attempts=0))
            d["url"] = m.group(3)
        m = RE_STAT.search(ln)
        if m:
            ev["stat"].append((m.group(1), m.group(2)))
        m = RE_S3L.search(ln)
        if m:
            ev["s3"].append((m.group(1), m.group(2), m.group(3), int(m.group(4))))
        m = RE_OPURI.search(ln)
        if m:
            ev["opuri"].append((m.group(1), m.group(2)))
        m = RE_LISTC.search(ln)
        if m:
            ev["listc"].append((m.group(1), m.group(2)))
        m = RE_CLIENT.search(ln)
        if m:
            ev["clients"].append(m.group(2))
    ev["sdk"] = list(inv.values())
    return ev


def parse_url(u):
    """'https://host[:port]/bucket' -> (scheme, host, port|None, bucket|None)."""
    m = re.match(r"^(\w+)://([^/:]+)(?::(\d+))?(?:/([^/]*))?", u or "")
    if not m:
        return None
    return (m.group(1), m.group(2).lower(), int(m.group(3)) if m.group(3) else None, m.group(4))


def server_tuple(s):
    """'https://h:443' | 'h:443' | 'h' -> (scheme|None, host, port|None)"""
    if not s:
        return None
    if "://" in s:
        p = parse_url(s)
        return (p[0], p[1], p[2]) if p else None
    m = re.match(r"^([^/:]+)(?::(\d+))?", s)
    return (None, m.group(1).lower(), int(m.group(2)) if m.group(2) else None)


def eff_port(t):
    if t[2] is not None:
        return t[2]
    return {"https": 443, "http": 80}.get(t[0])


def server_diff(asked, actual):
    """-> list of differing aspects ('host', 'port', 'scheme')"""
    a, b = server_tuple(asked), server_tuple(actual)
    if not a or not b:
        return []
    d = []
    if a[1] != b[1]:
        d.append("host")
    pa, pb = eff_port(a), eff_port(b)
    if pa is not None and pb is not None and pa != pb:
        d.append("port")
    if a[0] and b[0] and a[0] != b[0]:
        d.append("scheme")
    return d


def fmt_srv(t):
    return f"{t[0] or '?'}://{t[1]}" + (f":{t[2]}" if t[2] else "") if t else "-"


# --------------------------------------------------------------------------- op-log rows
def ts_ns(s):
    m = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?Z$", s or "")
    if not m:
        return None
    sec = calendar.timegm(__import__("time").strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))
    return sec * 10**9 + int((m.group(2) or "0").ljust(9, "0"))


def read_oplog(path):
    info = dict(path=path, exists=os.path.exists(path), rows=[], problems=[], size=None, header_ok=None)
    if not info["exists"]:
        info["problems"].append("op-log file does not exist")
        return info
    info["size"] = os.path.getsize(path)
    r = subprocess.run(["zstd", "-dc", path], capture_output=True)
    if r.returncode != 0:
        info["problems"].append("zstd -dc failed (truncated/unfinalized stream?): " + r.stderr.decode()[-120:].strip())
    text = r.stdout.decode("utf-8", "replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        info["problems"].append("empty op-log")
        return info
    info["header_ok"] = lines[0] == HEADER
    if not info["header_ok"]:
        info["problems"].append("header differs: " + lines[0][:100])
    keys = HEADER.split("\t")
    for i, ln in enumerate(lines[1:]):
        f = ln.split("\t")
        if len(f) != 13:
            info["problems"].append(f"row {i}: {len(f)} columns")
            continue
        d = dict(zip(keys, f))
        d["start_ns"], d["end_ns"] = ts_ns(d["start"]), ts_ns(d["end"])
        d["fb_ns"] = ts_ns(d["first_byte"]) if d["first_byte"] else None
        d["n"] = i
        try:
            d["bytes_i"], d["nobj_i"], d["dur_i"], d["idx_i"] = int(d["bytes"]), int(d["n_objects"]), int(d["duration_ns"]), int(d["idx"])
        except ValueError:
            info["problems"].append(f"row {i}: non-integer numeric column")
            continue
        info["rows"].append(d)
    # sanity
    for i, d in enumerate(info["rows"]):
        if d["idx_i"] != i:
            info["problems"].append(f"row {i}: idx {d['idx_i']} not contiguous")
            break
    for d in info["rows"]:
        if d["start_ns"] is None or d["end_ns"] is None:
            info["problems"].append(f"row idx {d['idx']}: unparsable start/end")
        elif d["end_ns"] < d["start_ns"]:
            info["problems"].append(f"row idx {d['idx']}: end before start")
        elif abs((d["end_ns"] - d["start_ns"]) - d["dur_i"]) > 1_000_000:
            info["problems"].append(f"row idx {d['idx']}: duration_ns {d['dur_i']} != end-start {d['end_ns'] - d['start_ns']}")
    return info


def decode_row(r, call):
    """Row -> what it says: kind, server, bucket, key (None = not recorded)."""
    ep, f = r["endpoint"], r["file"]
    if ep.startswith(("file://", "direct://")):
        return dict(kind="file", server=None, bucket=None, key=f)
    if re.match(r"^https?://", ep):
        return dict(kind="inner", server=ep, bucket=None, key=f)
    m = re.match(r"^s3://([^/]*)$", ep)
    if m:
        a = m.group(1)
        rest = f[len(a) + 1:] if f.startswith(a + "/") else f
        if call["asked"].get("hostform"):
            if "/" in rest:
                b, k = rest.split("/", 1)
            else:
                b, k = rest, ""
            return dict(kind="outer", server=a, bucket=b, key=k)
        return dict(kind="outer", server=None, bucket=a, key=rest)
    return dict(kind="other", server=None, bucket=None, key=f)


OPS_OK = {"GET": {"GET"}, "GET_RANGE": {"GET_RANGE"}, "PUT": {"PUT"}, "STAT": {"HEAD", "STAT"},
          "LIST": {"LIST"}, "DELETE": {"DELETE"}}


# --------------------------------------------------------------------------- native state helpers
def flat(snap, only=None):
    out = {}
    for sk, objs in (snap or {}).items():
        b = sk.split("|")[0]
        for n, v in objs.items():
            out[(b, n)] = v
    return out


def expected_listing(bucket, prefix, recursive, names):
    out = set()
    for n in names:
        if not n.startswith(prefix):
            continue
        rest = n[len(prefix):]
        if recursive or "/" not in rest:
            out.add(f"s3://{bucket}/{n}")
        else:
            out.add(f"s3://{bucket}/{prefix}{rest.split('/')[0]}/")
    return out


def short(s, n=60):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "~"


def sk(kind, items):
    return kind if not items else kind + ":" + ",".join(sorted(set(items)))


# --------------------------------------------------------------------------- per case
class Case:
    pass


def load_case(run, tag, fx, s0):
    cd = os.path.join(run, "cases", tag)
    c = Case()
    c.tag, c.dir = tag, cd
    c.ground = json.load(open(os.path.join(cd, "ground.json"))) if os.path.exists(os.path.join(cd, "ground.json")) else None
    c.native = json.load(open(os.path.join(cd, "native.json")))
    c.proc = c.native["proc"]
    c.segs, c.attempts, c.lines = parse_client_log(os.path.join(cd, "client.log"))
    opl = None
    if c.ground and c.ground.get("oplog_path") and not c.ground.get("final", {}).get("no_oplog"):
        opl = c.ground["oplog_path"]
        opl = os.path.join(cd, os.path.basename(opl))
    c.oplog = read_oplog(opl) if opl else None
    c.before = flat(c.native.get("snap_before"))
    c.after = flat(c.native.get("snap_after"))
    c.snap_scopes = set(c.native.get("snap_before", {}).keys())
    return c


def analyse_case(c, fx, s0, run_meta):
    res = []
    g = c.ground
    if not g:
        return res
    fxl = list(fx.items())
    by_size = {}
    for n, f in fxl:
        by_size.setdefault((f["bucket"], f["size"]), []).append(n)
    created = {k: v for k, v in c.after.items() if k not in c.before or c.before[k]["md5"] != v["md5"]}
    removed = {k: v for k, v in c.before.items() if k not in c.after}
    c.created, c.removed = created, removed
    c.stray_removed = []
    names_by_bucket = {}
    for (b, n) in flat(s0):
        names_by_bucket.setdefault(b, set()).add(n)
    base = dict(c.before) if c.before else {}
    # listing base: case's own before snapshot where it has one, else global S0
    list_base = {}
    for b, nset in names_by_bucket.items():
        list_base[b] = set(nset)
    for (b, n) in c.before:
        list_base.setdefault(b, set()).add(n)
    state_changes = []          # earlier PUT/DELETE asks in this case, applied to listing expectation
    rows_all = list(c.oplog["rows"]) if c.oplog else []
    used = set()
    pending_end = g.get("final", {}).get("t_ns") or 0
    for call in g["calls"]:
        o = call["outcome"]
        a = call["asked"]
        seg_id = call["group"] or call["id"]
        ev = seg_evidence(c.segs.get(seg_id, []))
        r = dict(case=c.tag, call=call, ev=ev, action=[], action_note=[], notes=[], actual={})
        ok = bool(o.get("ok"))
        hang = bool(o.get("hang"))
        # ---------------- ACTUAL: server / bucket (SDK log) ----------------
        urls = [parse_url(s["url"]) for s in ev["sdk"] if s["url"]]
        srvs = sorted({f"{u[0]}://{u[1]}" + (f":{u[2]}" if u[2] else "") for u in urls if u})
        bks = sorted({u[3] for u in urls if u and u[3]})
        act = dict(server=srvs[0] if srvs else None, servers=srvs, bucket=None, key=None, sdk_ops=[s["op"] for s in ev["sdk"]],
                   requests=sum(s["attempts"] for s in ev["sdk"]))
        ident = None          # fixture/object identity from content (strongest evidence for reads)
        # ---------------- ACTUAL: object ----------------
        k = call["kind"]
        got = None
        if ok and o.get("got_b64") is not None:
            got = base64.b64decode(o["got_b64"])
        if k in ("GET", "GET_RANGE") and ok:
            if "expect_sha256" in call and o.get("sha256") == call["expect_sha256"]:
                ident = (a["bucket"], a["key"])
            else:
                lo = call["args"].get("offset", 0)
                cands = []
                for n, f in fxl:
                    full = base64.b64decode(f["content_b64"])
                    seg = full[lo:lo + (call["args"].get("length") or len(full))] if k == "GET_RANGE" else full
                    if got is not None and seg == got:
                        cands.append((f["bucket"], f["key"], n))
                if cands:
                    ident = (cands[0][0], cands[0][1])
                    r["action_note"].append("bytes returned are those of fixture '%s' (%s/%s)" % (cands[0][2], cands[0][0], short(cands[0][1], 40)))
                else:
                    ident = ("?", "?")
                    r["action_note"].append("returned bytes match NO known object (len=%s)" % o.get("len"))
            act["bucket"], act["key"] = ident
        elif k == "STAT" and ok and call["api"] != "exists":
            sz = (o.get("stat") or {}).get("size") if isinstance(o.get("stat"), dict) else None
            cands = [n for (b, s), ns in by_size.items() for n in ns if s == sz]
            lk = ev["stat"][0] if ev["stat"] else None
            if sz is not None and cands:
                f = fx[cands[0]]
                ident = (f["bucket"], f["key"])
                if (a["bucket"], a["key"]) != ident:
                    r["action_note"].append("stat size %s is that of fixture '%s' (%s/%s)" % (sz, cands[0], f["bucket"], short(f["key"], 40)))
            if lk:
                r["action_note"].append("client log: stat_object bucket=%r key=%r" % (lk[0], short(lk[1], 50)))
                if ident is None or ident[1] != lk[1]:
                    ident = (lk[0], lk[1]) if ident is None else ident
            act["bucket"], act["key"] = ident if ident else (None, None)
        elif k == "STAT" and call["api"] == "exists":
            lk = ev["stat"][0] if ev["stat"] else None
            ident = (a["bucket"], a["key"])
            act["bucket"], act["key"] = ident
            if lk:
                act["bucket"], act["key"] = lk
        elif a.get("scheme") == "file" and k in ("PUT", "DELETE"):
            la = call.get("local_after")
            act["bucket"], act["key"] = None, a["key"]
            if ok and la is not None:
                if k == "PUT" and not (la["exists"] and la["sha256"] == a.get("put_sha256")):
                    r["no_payload"] = True
                    r["action_note"].append("file missing or different after put")
                if k == "DELETE" and la["exists"]:
                    r["no_payload"] = True
                    r["action_note"].append("file still exists after delete")
        elif k == "PUT":
            md5 = a.get("put_md5_b64")
            nat = {(x["bucket"], x["key"]): x for x in call.get("native_after", [])}
            r["unverified"] = False
            hit = [kk for kk, x in nat.items() if x["present"] and x["md5"] == md5]
            snaphit = [kk for kk, v in created.items() if v["md5"] == md5]
            if ok and (a["bucket"], a["key"]) in hit:
                act["bucket"], act["key"] = a["bucket"], a["key"]
            elif ok and (hit or snaphit):
                act["bucket"], act["key"] = (hit or snaphit)[0]
                r["action_note"].append("payload found at %s/%s instead" % (act["bucket"], short(act["key"], 50)))
            elif ok and "native_after" in call:
                act["bucket"], act["key"] = None, None
                r["action_note"].append("payload (md5) not found at the asked key nor at the watched alternatives")
                r["no_payload"] = True
            elif ok:
                r["unverified"] = True
            else:
                act["bucket"], act["key"] = a["bucket"], a["key"]
        elif k == "DELETE":
            nat = call.get("native_after", [])
            tgt = [x for x in nat if x["role"] == "target"]
            dec = [x for x in nat if x["role"] == "decoy"]
            r["unverified"] = False
            if ok and tgt:
                if not tgt[0]["present"]:
                    act["bucket"], act["key"] = tgt[0]["bucket"], tgt[0]["key"]
                else:
                    act["bucket"], act["key"] = None, None
                    r["action_note"].append("asked object is STILL PRESENT natively after delete returned success")
                    r["no_payload"] = True
                for x in dec:
                    if not x["present"]:
                        act["bucket"], act["key"] = x["bucket"], x["key"]
                        r["action_note"].append("DECOY object %s/%s was deleted instead/also" % (x["bucket"], short(x["key"], 50)))
                        r["decoy_deleted"] = True
            elif ok:
                r["unverified"] = True
            else:
                act["bucket"], act["key"] = a["bucket"], a["key"]
        elif k == "LIST":
            lc = ev["listc"][0] if ev["listc"] else None
            if lc:
                act["bucket"], act["key"] = lc
            else:
                act["bucket"], act["key"] = (bks[0] if bks else None), None
        if not ok and not hang and k in ("GET", "GET_RANGE", "STAT"):
            # failed read: the object the request named is only visible in the client log
            st_ = [x for x in ev["stat"] if (not call["group"]) or x[1] == a["key"]]
            s3_ = [x for x in ev["s3"] if (not call["group"]) or x[2] == a["key"]]
            lk = st_[0] if st_ else (s3_[0][1:3] if s3_ else None)
            act["bucket"], act["key"] = (lk if lk else (a["bucket"], a["key"]))
        r["actual"] = act
        # ---------------- ACTION verdict ----------------
        wa = []
        if a.get("scheme") == "file":                   # local files: key is an absolute path
            if ok and k in ("GET", "GET_RANGE") and "expect_sha256" in call and o.get("sha256") != call["expect_sha256"]:
                wa.append("object")
            if k == "LIST" and ok and "listing" in o and os.path.basename(g.get("oplog_path") or "") not in " ".join(o["listing"]):
                r["action_note"].append("local listing does not show the op-log file that sits in the directory")
                wa.append("listing")
            if call["expect_error"] and ok:
                wa.append("outcome")
        else:
            sd = server_diff(a.get("server"), act["server"]) if act["server"] else []
            for x in sd:
                wa.append(("server" if x == "host" else x))
            if act["server"] and len(srvs) > 1:
                r["notes"].append("several servers contacted: " + ", ".join(srvs))
            if k in ("GET", "GET_RANGE", "STAT", "PUT", "DELETE") and not (not ok and not hang and call["expect_error"] and k != "PUT"):
                if act["bucket"] not in (None, a["bucket"]) and act["bucket"] != "?":
                    wa.append("bucket")
                if act["key"] not in (None, a["key"]) and act["key"] != "?":
                    wa.append("key")
                if act["bucket"] == "?":
                    wa.append("object")
            elif k == "LIST":
                if act["bucket"] not in (None, a["bucket"]):
                    wa.append("bucket")
                if act["key"] not in (None, a["key"]):
                    wa.append("key")
            # SDK bucket (what the HTTP request addressed) vs asked bucket, for any op
            if bks and a.get("bucket") and bks[0] != a["bucket"] and "bucket" not in wa:
                wa.append("bucket")
            if "expect_sha256" in call and ok and o.get("sha256") != call["expect_sha256"] and not (act["bucket"], act["key"]) != (a["bucket"], a["key"]):
                wa.append("object")
            if call["expect_error"] and ok and not hang:
                if k in ("GET", "GET_RANGE", "STAT", "PUT"):
                    wa.append("server" if "server" not in wa and a.get("hostform") else "outcome")
                    r["action_note"].append("expected failure (nothing answers / object absent) but the call succeeded")
        if k == "LIST" and ok and "listing" in o and a.get("scheme") != "file":
            names = set(list_base.get(a["bucket"], set()))
            for st in state_changes:
                if st[0] == "PUT" and st[1] == a["bucket"]:
                    names.add(st[2])
                if st[0] == "DELETE" and st[1] == a["bucket"]:
                    names.discard(st[2])
            exp = expected_listing(a["bucket"], a["key"], bool(call["args"].get("recursive")), names)
            gotl = set(o["listing"])
            if gotl != exp:
                wa.append("listing")
                r["action_note"].append("listing differs: missing=%s extra=%s" % (sorted(exp - gotl)[:3], sorted(gotl - exp)[:3]))
            r["list_expected_n"] = len(exp)
        if k == "GET" and ok and o.get("len") is not None and "expect_len" in call and o["len"] != call["expect_len"] and "object" not in wa:
            wa.append("object")
        if k == "GET_RANGE" and ok and "expect_len" in call and o.get("len") != call["expect_len"]:
            wa.append("range")
        if k in ("PUT", "DELETE") and ok and call["api"] != "delete_batch(rust)":
            state_changes.append((k, act["bucket"] or a["bucket"], act["key"] or a["key"]))
        if k == "STAT" and ok and call["api"] == "exists" and call["args"].get("exists") is False and o.get("exists"):
            wa.append("outcome")
        if r.get("no_payload"):
            wa.append("object")
        r["wa"] = sorted(set(wa))
        if r.get("unverified"):
            wa_sig = "UNVERIFIED"
        else:
            wa_sig = None
        if hang:
            r["action_sig"] = "HANG" if not wa else "HANG+WRONG-ACTION:" + ",".join(r["wa"])
        else:
            r["action_sig"] = wa_sig or ("OK" if not wa else "WRONG-ACTION:" + ",".join(r["wa"]))
        # ---------------- RECORD verdict ----------------
        r["rows"] = []
        r["rec_fields"] = []
        r["rec_note"] = []
        if c.oplog is None:
            r["record_sig"] = "N/A"
            r["rec_note"].append("no op-log in this probe")
        else:
            t0 = call["t_start_ns"] - 5_000_000
            t1 = (pending_end if hang else call["t_end_ns"]) + 5_000_000
            win = [d for d in rows_all if d["start_ns"] is not None and t0 <= d["start_ns"] <= t1 and d["n"] not in used]
            if call["group"]:
                mine = [d for d in win if d["file"].endswith(a["key"]) or d["file"] == a["key"]]
            else:
                mine = win
            for d in mine:
                used.add(d["n"])
            r["rows"] = mine
            fields = set()
            for d in mine:
                dr = decode_row(d, call)
                d["_dec"] = dr
                f = row_fields(d, dr, call, act, o, ok, hang)
                d["_bad"] = f
                fields |= set(f)
                if CRED.match(d["client_id"] or ""):
                    r["rec_note"].append("client_id carries a credential (row idx %s)" % d["idx"])
                if dr["server"] is None and dr["kind"] == "outer":
                    r["rec_note"].append("row idx %s: server not recorded" % d["idx"])
                if dr["bucket"] is None and dr["kind"] == "inner":
                    r["rec_note"].append("row idx %s: bucket not recorded" % d["idx"])
            r["rec_fields"] = sorted(fields)
            n = len(mine)
            if hang:
                r["record_sig"] = "HANG" if n == 0 else "HANG+ROWS"
            elif n == 0:
                r["record_sig"] = "MISSING"
            elif n > 1:
                r["record_sig"] = "DUPLICATE" + ("+WRONG-FIELD:" + ",".join(sorted(fields)) if fields else "")
            elif fields:
                r["record_sig"] = "WRONG-FIELD:" + ",".join(sorted(fields))
            else:
                r["record_sig"] = "OK"
        res.append(r)
    # phantom rows (not attributed to any call)
    c.phantom = [d for d in rows_all if d["n"] not in used]
    return res


def row_fields(d, dr, call, act, o, ok, hang):
    """Which fields of this row disagree with what ACTUALLY happened."""
    bad = []
    k = call["kind"]
    a = call["asked"]
    if d["op"] not in OPS_OK.get(k, {k}):
        bad.append("op")
    # server
    if dr["server"] is not None and act["server"]:
        if server_diff(dr["server"], act["server"]):
            bad.append("server")
    elif dr["server"] is not None and call["asked"].get("hostform") and not act["server"]:
        pass
    # bucket
    if dr["bucket"] is not None and act["bucket"] not in (None, "?") and dr["bucket"] != act["bucket"]:
        bad.append("bucket")
    # key (exact, leading slash matters)
    if dr["key"] is not None and act["key"] not in (None, "?"):
        want = act["key"] if dr["kind"] != "file" else a["key"]
        if dr["key"] != want:
            bad.append("key")
    elif dr["key"] is not None and act["key"] is None and dr["kind"] == "file":
        if dr["key"] != a["key"]:
            bad.append("key")
    # error
    row_err = d["error"] != ""
    want_err = (not ok) or (call["api"] == "exists" and o.get("exists") is False)
    if not hang and row_err != want_err:
        bad.append("error")
    # bytes / n_objects
    if ok and not hang:
        if k in ("GET", "GET_RANGE"):
            if o.get("len") is not None and d["bytes_i"] != o["len"]:
                bad.append("bytes")
        elif k == "PUT":
            if "put_len" in a and d["bytes_i"] != a["put_len"]:
                bad.append("bytes")
        elif k == "STAT" and call["api"] != "exists":
            sz = (o.get("stat") or {}).get("size") if isinstance(o.get("stat"), dict) else None
            if d["bytes_i"] not in (0, sz):
                bad.append("bytes")
        elif k in ("LIST", "DELETE"):
            if d["bytes_i"] != 0:
                bad.append("bytes")
        if k == "LIST":
            if "listing" in o and d["nobj_i"] != len(o["listing"]):
                bad.append("n_objects")
        elif d["nobj_i"] != 1:
            bad.append("n_objects")
    return bad


# --------------------------------------------------------------------------- known defects
def glob_any(value, pat):
    pats = pat if isinstance(pat, list) else [pat]
    return any(fnmatch.fnmatch(value, p) for p in pats)


def find_known(known, case_tag, call):
    case = case_tag.split("__")[0]
    form = case_tag.split("__")[1] if "__" in case_tag else ""
    for kn in known["expected"]:
        m = kn["match"]
        if (glob_any(case, m.get("case", "*")) and glob_any(call["id"], m.get("call", "*"))
                and glob_any(call["api"], m.get("api", "*")) and glob_any(form, m.get("form", "*"))):
            return kn
    return None


def classify(r, known):
    kn = find_known(known, r["case"], r["call"])
    a, rc = r["action_sig"], r["record_sig"]
    if kn is None:
        if a == "OK" and rc in ("OK", "N/A"):
            return "PASS", None
        return "NEW", None
    ea, er = kn["action"], kn["record"]
    if a == ea and rc == er:
        if a == "OK" and rc in ("OK", "N/A"):
            return "PASS", kn
        return "KNOWN", kn
    if a == "OK" and rc in ("OK", "N/A"):
        return "FIXED", kn
    return "CHANGED", kn


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--known", default=os.path.join(HERE, "known.json"))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--all-details", action="store_true", help="print the details block for PASS calls too")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    meta = json.load(open(os.path.join(run, "run.json")))
    fxd = json.load(open(os.path.join(run, "fixtures.json")))
    fx = fxd["fixtures"]
    snaps = json.load(open(os.path.join(run, "snapshots.json"))) if os.path.exists(os.path.join(run, "snapshots.json")) else {}
    s0 = snaps.get("S0", {})
    known = json.load(open(a.known)) if os.path.exists(a.known) else dict(expected=[], process=[])
    tags = sorted(os.listdir(os.path.join(run, "cases")))
    order = [x["tag"] for x in meta.get("case_results", [])] or tags
    out = []
    P = out.append
    allres = []
    cases = []
    total_att = 0
    for tag in order:
        if not os.path.isdir(os.path.join(run, "cases", tag)):
            continue
        c = load_case(run, tag, fx, s0)
        c.results = analyse_case(c, fx, s0, meta)
        total_att += c.attempts
        cases.append(c)
        allres += c.results

    # ---------------------------------------------------------------- table
    P("=" * 150)
    P(f"TRACE-ACCURACY REPORT   s3dlio {meta.get('s3dlio_version')}   wheel {os.path.basename(meta['wheel'])}   prefix {meta['prefix']}")
    P(f"truth: ground-truth call list + gcloud native state + s3dlio client debug log (GCS S3 endpoint {meta['endpoint']}); no local server, no consumer tools")
    P("=" * 150)
    hdr = f"{'case/form':26} {'call':18} {'api':16} {'target (as asked)':44} | {'(1) ACTION':26} | {'(2) RECORD':30} | status"
    P(hdr)
    P("-" * len(hdr))
    status_count = {}
    for c in cases:
        for r in c.results:
            call = r["call"]
            st, kn = classify(r, known)
            r["status"], r["known"] = st, kn
            status_count[st] = status_count.get(st, 0) + 1
            a_ = call["asked"]
            tgt = f"{a_.get('bucket') or ''}/{short((a_.get('key') or '').replace(meta['prefix'], '<P>'), 30)}" if a_.get("scheme") != "file" else "file:" + short(os.path.basename(a_["key"]) or "dir/", 30)
            if a_.get("hostform"):
                tgt = short(a_["server"].replace("https://", ""), 22) + "|" + tgt
            tag = f"{(kn['issues'] and ' ' + '/'.join(kn['issues'])) if kn else ''}"
            P(f"{c.tag:26} {call['id']:18} {call['api']:16} {short(tgt, 44):44} | {short(r['action_sig'], 26):26} | {short(r['record_sig'], 30):30} | {st}{tag}")
    P("")

    # ---------------------------------------------------------------- details
    P("=" * 150)
    P("DETAILS (every call whose verdict is not PASS)")
    P("=" * 150)
    for c in cases:
        for r in c.results:
            if r["status"] == "PASS" and not a.all_details:
                continue
            call, o = r["call"], r["call"]["outcome"]
            P(f"[{c.tag} :: {call['id']}] {call['api']}  {short(call['uri'].replace(meta['prefix'], '<P>'), 110)}   -> {r['status']}"
              + (f"  (known {','.join(r['known']['issues'])}: expected action={r['known']['action']} record={r['known']['record']})" if r["known"] else ""))
            P(f"    outcome: " + ("HANG (no return within %ss)" % os.environ.get("H_CALL_TIMEOUT", "call timeout") if o.get("hang") else ("ok" if o.get("ok") else "ERROR: " + short(o.get("error"), 140))))
            act = r["actual"]
            P(f"    asked : server={call['asked'].get('server')} bucket={call['asked'].get('bucket')} key={short(str(call['asked'].get('key')).replace(meta['prefix'], '<P>'), 70)}")
            P(f"    actual: server={','.join(act['servers']) or '-'} bucket={act['bucket']} key={short(str(act['key']).replace(meta['prefix'], '<P>'), 70)}  sdk_requests={act['requests']} ({','.join(act['sdk_ops'])})")
            for n in r["action_note"]:
                P("    evidence: " + short(n.replace(meta["prefix"], "<P>"), 200))
            for d in r["rows"]:
                dr = d["_dec"]
                cid = "<credential>" if CRED.match(d["client_id"] or "") else (d["client_id"] or "''")
                P("    row %s: %-9s client_id=%s n=%s bytes=%s endpoint=%s file=%s err=%s  -> %s" % (
                    d["idx"], d["op"], cid, d["nobj_i"], d["bytes_i"], d["endpoint"], short(d["file"].replace(meta["prefix"], "<P>"), 70),
                    short(d["error"], 30) or "''", ("bad: " + ",".join(d["_bad"])) if d["_bad"] else "fields ok"))
            if not r["rows"] and c.oplog is not None:
                P("    row   : none in window")
            for n in sorted(set(r["rec_note"])):
                P("    note  : " + n)
            for n in r["notes"]:
                P("    note  : " + n)
        # process / lifecycle / phantom
    # ---------------------------------------------------------------- per-case lifecycle
    P("")
    P("=" * 150)
    P("PER-CASE LIFECYCLE / FILE CHECKS")
    P("=" * 150)
    anomalies = []
    for c in cases:
        g = c.ground or {}
        items = []
        pr = c.proc
        if pr["rc"] != 0 or pr["timed_out"] or pr["stuck"]:
            kproc = [k for k in known.get("process", []) if glob_any(c.tag.split("__")[0], k["case"])]
            items.append(("PROCESS", f"exit rc={pr['rc']} timed_out={pr['timed_out']} stuck={pr['stuck']}", kproc[0] if kproc else None))
        for i in g.get("inits", []):
            if i.get("arg", "").startswith("s3://"):
                bad = []
                if i.get("ok"):
                    bad.append("s3:// destination was ACCEPTED")
                if i.get("active_after"):
                    bad.append("op-log active after the rejected init")
                if i.get("s3_dir_created") or i.get("local_listing_after") != i.get("local_listing_before"):
                    bad.append("a local file/dir was created: %s" % sorted(set(i.get("local_listing_after", [])) - set(i.get("local_listing_before", []))))
                nd = i.get("native_dest") or []
                if any(x.get("present") for x in nd):
                    bad.append("an object named by the s3:// destination exists natively")
                if c.created or c.removed:
                    bad.append("native state changed: created=%s removed=%s" % (sorted(c.created), sorted(c.removed)))
                items.append(("S3-DEST-REJECT", "OK: rejected (%s); nothing created locally or natively" % short(i.get("error", ""), 80) if not bad else "WRONG: " + "; ".join(bad), None))
            elif not i.get("ok"):
                items.append(("INIT", "init_op_log failed: " + short(i.get("error"), 120), None))
        fin = g.get("final", {})
        if fin and not fin.get("no_oplog"):
            if not fin.get("finalize_ok") or fin.get("finalize_hung"):
                items.append(("FINALIZE", "finalize_op_log failed/hung: " + short(str(fin), 160), None))
            elif fin.get("active_after"):
                items.append(("FINALIZE", "is_op_log_active() still True after finalize", None))
        elif not fin and g:
            items.append(("FINALIZE", "case never reached finalize (process died before finish)", None))
        if c.oplog is not None:
            for p in c.oplog["problems"][:5]:
                items.append(("OPLOG-FILE", p, None))
            if c.tag.startswith("local_files"):
                selfrows = [d for d in c.oplog["rows"] if os.path.basename(c.oplog.get("path", "")) in d["file"]]
                items.append(("SELF-RECORD", "OK: op-log never recorded itself" if not selfrows else "WRONG: %d row(s) name the op-log file" % len(selfrows), None))
        if getattr(c, "phantom", None):
            for d in c.phantom:
                items.append(("PHANTOM-ROW", "row idx %s %s %s matches no call of this case" % (d["idx"], d["op"], short(d["file"].replace(meta["prefix"], "<P>"), 60)), None))
        if getattr(c, "stray_removed", None):
            items.append(("STRAY-DELETE", "objects disappeared that no delete asked for: %s" % c.stray_removed, None))
        if items:
            P(f"{c.tag}:")
            for kind, text, kn in items:
                tagk = ""
                if kn:
                    tagk = f"   [KNOWN {','.join(kn['issues'])}: {kn.get('note', '')}]"
                    anomalies.append((c.tag, kind, text, "KNOWN"))
                elif kind not in ("S3-DEST-REJECT", "SELF-RECORD") or text.startswith("WRONG"):
                    anomalies.append((c.tag, kind, text, "NEW"))
                P(f"    {kind:15} {text}{tagk}")

    # ---------------------------------------------------------------- stray objects (final audit)
    fin = flat(snaps.get("FINAL", {}))
    s0f = flat(s0)
    expected_new = set()
    for c in cases:
        if c.ground:
            for call in c.ground["calls"]:
                if call["kind"] == "PUT" and call["asked"].get("scheme") != "file":
                    expected_new.add((call["asked"]["bucket"], call["asked"]["key"]))
    stray = sorted(k for k in fin if k not in s0f and k not in expected_new)
    gone = sorted(k for k in s0f if k not in fin)
    P("")
    P("FINAL NATIVE AUDIT (state when the run ended, before cleanup): objects at the end that no PUT asked for: %s" % [(b, n.replace(meta['prefix'], '<P>')) for b, n in stray])
    asked_del = set()
    for c in cases:
        if c.ground:
            for call in c.ground["calls"]:
                if call["kind"] == "DELETE":
                    asked_del.add((call["asked"]["bucket"], call["asked"]["key"]))
    unexpected_gone = [x for x in gone if x not in asked_del]
    P("   fixtures no longer present at the end: %s" % [(b, n.replace(meta['prefix'], '<P>')) for b, n in gone])
    P("   ... of which NO delete call asked for: %s" % [(b, n.replace(meta['prefix'], '<P>')) for b, n in unexpected_gone])
    kn_gone = []
    for x in list(unexpected_gone):
        for ka in known.get("audit_unasked_deletions", []):
            if x == ({"B1": meta["b1"], "B2": meta["b2"]}.get(ka["bucket"], ka["bucket"]), meta["prefix"] + "/" + ka["key_rel"]):
                kn_gone.append((x, ka))
                unexpected_gone.remove(x)
    for x, ka in kn_gone:
        P("   KNOWN %s: %s/%s deleted although no call asked for it (%s)" % (",".join(ka["issues"]), x[0], x[1].replace(meta["prefix"], "<P>"), ka["note"]))
    if stray or unexpected_gone:
        anomalies.append(("FINAL-AUDIT", "state", "stray objects %s / unasked deletions %s" % (stray, unexpected_gone), "NEW"))

    # ---------------------------------------------------------------- summary
    P("")
    P("=" * 150)
    P("SUMMARY (calls; 'N-of-total')")
    P("=" * 150)

    def count(rs, pred):
        return sum(1 for r in rs if pred(r))

    def line(label, rs):
        n = len(rs)
        ao = count(rs, lambda r: r["action_sig"] == "OK")
        aw = count(rs, lambda r: r["action_sig"].startswith("WRONG-ACTION") or "WRONG-ACTION" in r["action_sig"])
        ah = count(rs, lambda r: r["action_sig"].startswith("HANG") and "WRONG-ACTION" not in r["action_sig"])
        ro = count(rs, lambda r: r["record_sig"] == "OK")
        rm = count(rs, lambda r: r["record_sig"] == "MISSING")
        rd = count(rs, lambda r: r["record_sig"].startswith("DUPLICATE"))
        rf = count(rs, lambda r: r["record_sig"].startswith("WRONG-FIELD") or "+WRONG-FIELD" in r["record_sig"])
        rh = count(rs, lambda r: r["record_sig"].startswith("HANG"))
        rn = count(rs, lambda r: r["record_sig"] == "N/A")
        st = lambda s: count(rs, lambda r: r["status"] == s)
        return (f"{label:26} calls={n:3} | ACTION ok={ao:3} wrong={aw:3} hang={ah:2} | RECORD ok={ro:3} missing={rm:3} duplicate={rd:3} wrong-field={rf:3} hang={rh:2} n/a={rn:2}"
                f" | PASS={st('PASS')} KNOWN={st('KNOWN')} NEW={st('NEW')} CHANGED={st('CHANGED')} FIXED={st('FIXED')}")
    for c in cases:
        P(line(c.tag, c.results))
    P("-" * 150)
    P(line("ALL CALLS", allres))
    phantom = sum(len(getattr(c, "phantom", [])) for c in cases)
    P(f"phantom op-log rows (rows with no call): {phantom}")
    P("")
    # NEW / CHANGED listing
    bad = [r for r in allres if r["status"] in ("NEW", "CHANGED")]
    fixed = [r for r in allres if r["status"] == "FIXED"]
    P(f"UNEXPECTED (NEW or CHANGED vs known.json): {len(bad)} of {len(allres)} calls")
    for r in bad:
        kn = r["known"]
        P(f"   {r['case']} :: {r['call']['id']:16} action={r['action_sig']}  record={r['record_sig']}" + (f"   (known said action={kn['action']} record={kn['record']})" if kn else "   (no known entry: expected OK/OK)"))
    P(f"KNOWN DEFECT NO LONGER PRESENT (FIXED): {len(fixed)}")
    for r in fixed:
        P(f"   {r['case']} :: {r['call']['id']:16} was {r['known']['issues']}")
    newproc = [x for x in anomalies if x[3] == "NEW"]
    P(f"UNEXPECTED case-level findings (process/lifecycle/file/phantom): {len(newproc)}")
    for t, k, text, _ in newproc:
        P(f"   {t}: {k}: {short(text, 160)}")
    for s in meta.get("skipped", []):
        P(f"SKIPPED CASE {s['case']}: {s['reason']}   (verdict SKIP: not evaluated, not a failure)")
    # which known entries were not exercised
    exercised = set()
    for r in allres:
        if r["known"]:
            exercised.add(id(r["known"]))
    P("")
    P("NOT COVERED by this harness (by design): create_bucket / delete_bucket (never create or delete GCS buckets); "
      "delete_batch op-log rows (the Rust probe has no logger; action checked only); GET_RANGE offset/length are not a column in the row format (cannot be verified from the trace).")
    # requests
    led = os.path.join(run, "requests.jsonl")
    gc = 0
    by = {}
    if os.path.exists(led):
        for ln in open(led):
            e = json.loads(ln)
            gc += e["n"]
            by[e["who"]] = by.get(e["who"], 0) + e["n"]
    P(f"REQUESTS: s3dlio SDK attempts (from client debug logs) = {total_att};  gcloud native = {gc} {by};  total = {total_att + gc}")
    cl = meta.get("cleanup", {})
    P(f"CLEANUP: deleted {cl.get('deleted')} objects; prefix empty in both buckets = {cl.get('empty')}; remaining = {cl.get('remaining')}")
    P(f"RUNTIME: {meta.get('elapsed_s')} s")
    txt = "\n".join(out)
    open(os.path.join(run, "report.txt"), "w").write(txt + "\n")
    slim = []
    for r in allres:
        slim.append(dict(case=r["case"], call=r["call"]["id"], api=r["call"]["api"], action=r["action_sig"], record=r["record_sig"],
                         status=r["status"], issues=(r["known"] or {}).get("issues"), rows=len(r["rows"])))
    json.dump(dict(calls=slim, status_count=status_count, requests=dict(sdk=total_att, gcloud=gc)),
              open(os.path.join(run, "results.json"), "w"), indent=1)
    if not a.quiet:
        print(txt)
    # exit status: 0 = nothing unexpected
    return 1 if (bad or newproc) else 0


if __name__ == "__main__":
    sys.exit(main())
