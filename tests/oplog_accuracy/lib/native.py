"""Native (Google) tooling wrapper: gcloud storage.  Every call is counted from `--log-http`.

Truth about object state comes ONLY from here (never from s3dlio, never from a local server).
Only method+uri lines of the http log are kept; credentials are never read or printed.
"""
import json
import os
import subprocess
import threading

_lock = threading.Lock()


def _ledger(entry):
    p = os.environ.get("H_REQ_LEDGER")
    if p:
        with _lock, open(p, "a") as f:
            f.write(json.dumps(entry) + "\n")


def gcx(args, who="gcloud", check=True, stdin=None):
    """Run `gcloud storage ...` -> (rc, stdout, stderr_without_http_log). Counts http requests."""
    cmd = ["gcloud"] + list(args) + ["--log-http"]
    r = subprocess.run(cmd, capture_output=True, text=True, input=stdin, timeout=180)
    reqs, other, cur = [], [], None
    for ln in r.stderr.splitlines():
        if ln.startswith("==== request start ===="):
            cur = {}
            reqs.append(cur)
        elif cur is not None and ln.startswith("uri: "):
            cur["uri"] = ln[5:].split("?")[0]
        elif cur is not None and ln.startswith("method: "):
            cur["method"] = ln[8:]
        elif ln.startswith("==== request end ===="):
            cur = None
        elif cur is None and not ln.startswith(("==== ", "---", "-- ")):
            other.append(ln)
    _ledger(dict(who=who, cmd=" ".join(args)[:200], n=len(reqs), reqs=[f"{q.get('method')} {q.get('uri')}" for q in reqs]))
    if check and r.returncode != 0:
        raise RuntimeError(f"gcloud {' '.join(args)[:120]} failed rc={r.returncode}: " + "\n".join(other[-5:]))
    return r.returncode, r.stdout, "\n".join(other)


def list_scope(bucket, name_prefix):
    """Objects whose NAME starts with name_prefix (may start with '/'), in `bucket`.
    -> {name: dict(size, md5, generation)}"""
    rc, out, err = gcx(["storage", "objects", "list", f"gs://{bucket}/{name_prefix}**", "--format=json"],
                       who="snapshot", check=False)
    if rc != 0:
        if "One or more URLs matched no objects" in err or "matched no objects" in err:
            return {}
        raise RuntimeError(f"list {bucket}/{name_prefix} rc={rc}: {err[-300:]}")
    items = json.loads(out) if out.strip() else []
    return {i["name"]: dict(size=int(i["size"]), md5=i.get("md5_hash"), generation=i.get("generation")) for i in items}


def describe(bucket, name):
    rc, out, err = gcx(["storage", "objects", "describe", f"gs://{bucket}/{name}", "--format=json"],
                       who="describe", check=False)
    if rc != 0:
        return None
    d = json.loads(out)
    return dict(size=int(d["size"]), md5=d.get("md5_hash"), generation=d.get("generation"))


def cat(bucket, name):
    r = subprocess.run(["gcloud", "storage", "cat", f"gs://{bucket}/{name}", "--log-http"], capture_output=True, timeout=120)
    n = r.stderr.count(b"==== request start ====")
    _ledger(dict(who="cat", cmd=f"cat gs://{bucket}/{name}", n=n))
    return r.stdout if r.returncode == 0 else None


def scopes(prefix, b1, b2):
    """The three name-spaces the harness is allowed to touch."""
    return [(b1, prefix + "/"), (b1, "/" + prefix + "/"), (b2, prefix + "/")]


def snapshot(prefix, b1, b2, which=None):
    snap = {}
    for sc in scopes(prefix, b1, b2):
        if which is None or sc in which:
            snap[f"{sc[0]}|{sc[1]}"] = list_scope(*sc)
    return snap


def rm_names(bucket, names):
    if not names:
        return
    gcx(["storage", "rm"] + [f"gs://{bucket}/{n}" for n in names], who="cleanup", check=False)
