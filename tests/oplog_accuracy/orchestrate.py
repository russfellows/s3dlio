"""Orchestrator: prefix, native setup, one fresh bounded process per case, native snapshots,
cleanup + proof.  Stdlib only; invoked by run.sh through `uv run --no-project --isolated`.

SIGTERM only, never SIGKILL: a case that does not exit after SIGTERM is reported as STUCK.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
import hlib      # noqa: E402
import native    # noqa: E402

ENV_B1, ENV_B2, ENV_HOST = "OPLOG_HARNESS_BUCKET1", "OPLOG_HARNESS_BUCKET2", "OPLOG_HARNESS_HOSTFORM_HOST"
SCOPE_NAMES = {"b1": 0, "b1lead": 1, "b2": 2}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def run_case(uvcmd, case, form, cdir, env, timeout):
    """Fresh process, new session, bounded.  Returns dict(rc, timed_out, stuck, secs)."""
    os.makedirs(cdir, exist_ok=True)
    cenv = dict(env, H_CASE_DIR=cdir, H_OPLOG_FORM=form, H_CASE=case["case"])
    cenv.update({k: str(v) for k, v in case.get("env", {}).items()})
    script = os.path.join(HERE, "cases", case["script"])
    t0 = time.time()
    with open(os.path.join(cdir, "client.log"), "wb") as lf:
        p = subprocess.Popen(uvcmd + [script], stdout=lf, stderr=subprocess.STDOUT, env=cenv,
                             cwd=cdir, start_new_session=True, stdin=subprocess.DEVNULL)
        timed_out = stuck = False
        try:
            p.wait(timeout=timeout)
        except BaseException as ex:        # noqa  (SIGTERM/INT of the orchestrator: pass TERM on, never KILL)
            if not isinstance(ex, subprocess.TimeoutExpired):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                    p.wait(timeout=30)
                except Exception:   # noqa
                    pass
                raise
            timed_out = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)      # second TERM; never KILL
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    stuck = True
    return dict(rc=p.returncode, timed_out=timed_out, stuck=stuck, secs=round(time.time() - t0, 2), pid=p.pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheel", required=True)
    ap.add_argument("--out", required=True, help="run directory (must not exist)")
    ap.add_argument("--cases", default="", help="comma list of case names (default all)")
    ap.add_argument("--all-forms", action="store_true", help="run EVERY case in both op-log forms")
    ap.add_argument("--delbatch-bin", default="", help="optional Rust probe for the delete_batch case (skipped when absent)")
    ap.add_argument("--no-cleanup", action="store_true")
    ap.add_argument("--python", default="3.13", help="interpreter version passed to uv (must match the wheel tag)")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))   # run the cleanup in `finally`
    # fail fast, before anything is created anywhere: configuration comes ONLY from the environment
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", ENV_B1, ENV_B2):
        if not os.environ.get(k):
            sys.exit(f"required environment variable {k} is not set (see README.md); nothing was created")
    B1, B2 = os.environ[ENV_B1], os.environ[ENV_B2]
    if B1 == B2:
        sys.exit(f"{ENV_B1} and {ENV_B2} must name two DIFFERENT buckets; nothing was created")
    hostform_host = os.environ.get(ENV_HOST) or urllib.parse.urlparse(os.environ["AWS_ENDPOINT_URL"]).hostname
    if not hostform_host:
        sys.exit(f"cannot derive a host from AWS_ENDPOINT_URL; set {ENV_HOST}; nothing was created")
    wheel = os.path.abspath(a.wheel)
    run = os.path.abspath(a.out)
    os.makedirs(run)
    today = datetime.date.today().strftime("%Y%m%d")
    rnd = "%06x" % random.randrange(1 << 24)
    prefix = f"s3dlio-harness-{today}-{rnd}"
    seed = "%08x" % random.randrange(1 << 32)
    ledger = os.path.join(run, "requests.jsonl")
    os.environ["H_REQ_LEDGER"] = ledger
    t_run0 = time.time()
    meta = dict(prefix=prefix, seed=seed, b1=B1, b2=B2, wheel=wheel, wheel_sha256=sha_file(wheel),
                started=datetime.datetime.now().isoformat(timespec="seconds"),
                endpoint=os.environ["AWS_ENDPOINT_URL"], hostform_host=hostform_host, delbatch_bin=a.delbatch_bin or None)
    fx = hlib.fixture_table(prefix, B1, B2, seed)
    json.dump(dict(meta=meta, fixtures=fx), open(os.path.join(run, "fixtures.json"), "w"), indent=1)

    manifest = json.load(open(os.path.join(HERE, "cases", "manifest.json")))
    want = [c for c in a.cases.split(",") if c]
    cases = [c for c in manifest if not want or c["case"] in want]
    skipped = []
    if not (a.delbatch_bin and os.path.isfile(a.delbatch_bin)):
        for c in cases:
            if c["case"] == "delete_batch":
                skipped.append(dict(case="delete_batch", reason="SKIP: no delete_batch probe given (--delbatch-bin missing or not a file)"))
        cases = [c for c in cases if c["case"] != "delete_batch"]
    meta["skipped"] = skipped
    for s in skipped:
        print(f"[skip] {s['case']}: {s['reason']}", flush=True)

    uvbase = ["uv", "run", "--no-project", "--isolated", "--python", a.python, "--with", wheel, "python", "-I"]
    # s3dlio version straight from the wheel under test
    r = subprocess.run(uvbase + ["-c", "import s3dlio;print(s3dlio.__version__)"], capture_output=True, text=True)
    meta["s3dlio_version"] = r.stdout.strip()
    log(f"prefix={prefix} seed={seed} s3dlio={meta['s3dlio_version']} wheel={os.path.basename(wheel)}")

    snaps = {}
    cleanup = {}
    try:
        # ------------------------------------------------------------ native setup
        loc = os.path.join(run, "fixtures-local")
        for name, f in fx.items():
            if f["key"].startswith("/"):
                continue
            bd = "b1" if f["bucket"] == B1 else "b2"
            pth = os.path.join(loc, bd, f["key"])
            os.makedirs(os.path.dirname(pth), exist_ok=True)
            open(pth, "wb").write(hlib.make_content(name, f["size"], seed))
        for bd, b in (("b1", B1), ("b2", B2)):
            native.gcx(["storage", "cp", "-r", os.path.join(loc, bd, prefix), f"gs://{b}/"], who="setup")
        for name, f in fx.items():
            if f["key"].startswith("/"):
                pth = os.path.join(loc, "lead", name)
                os.makedirs(os.path.dirname(pth), exist_ok=True)
                open(pth, "wb").write(hlib.make_content(name, f["size"], seed))
                native.gcx(["storage", "cp", pth, f"gs://{f['bucket']}/{f['key']}"], who="setup")
        s0 = native.snapshot(prefix, B1, B2)
        snaps["S0"] = s0
        allobj = {}
        for sk, objs in s0.items():
            b = sk.split("|")[0]
            for n, v in objs.items():
                allobj[(b, n)] = v
        bad = []
        for name, f in fx.items():
            got = allobj.get((f["bucket"], f["key"]))
            if not got or got["size"] != f["size"]:
                bad.append(name)
        extra = set(allobj) - {(f["bucket"], f["key"]) for f in fx.values()}
        if bad or extra:
            raise RuntimeError(f"native setup mismatch: bad={bad} extra={sorted(extra)}")
        log(f"setup ok: {len(fx)} objects natively verified (size) in {B1}/{B2}")

        # ------------------------------------------------------------ cases
        env = dict(os.environ, H_PREFIX=prefix, H_BUCKET1=B1, H_BUCKET2=B2, H_SEED=seed,
                   H_LIB=os.path.join(HERE, "lib"), H_DELBATCH_BIN=a.delbatch_bin, H_HOSTFORM_HOST=hostform_host)
        results = []
        for c in cases:
            forms = c.get("forms", ["path"]) if not a.all_forms else ["path", "fileuri"]
            if c.get("single_form"):
                forms = c["forms"]
            for form in forms:
                tag = f"{c['case']}__{form}"
                cdir = os.path.join(run, "cases", tag)
                scs = [list(native.scopes(prefix, B1, B2))[SCOPE_NAMES[s]] for s in c.get("mutates", [])]
                before = native.snapshot(prefix, B1, B2, which=scs) if scs else {}
                log(f"RUN {tag}")
                res = run_case(uvbase, c, form, cdir, env, c.get("timeout", 120))
                after = native.snapshot(prefix, B1, B2, which=scs) if scs else {}
                json.dump(dict(case=c["case"], form=form, tag=tag, proc=res, mutates=c.get("mutates", []),
                               snap_before=before, snap_after=after, manifest=c),
                          open(os.path.join(cdir, "native.json"), "w"), indent=1)
                results.append(dict(tag=tag, **res))
                log(f"  -> rc={res['rc']} secs={res['secs']} timed_out={res['timed_out']} stuck={res['stuck']}")
        # ------------------------------------------------------------ final state
        meta["case_results"] = results
    finally:
        # ------------------------------------------------------------ cleanup + proof
        if not a.no_cleanup:
            try:
                names = {}
                fin = native.snapshot(prefix, B1, B2)          # FINAL state = what the cleanup is about to remove
                snaps["FINAL"] = fin
                json.dump(snaps, open(os.path.join(run, "snapshots.json"), "w"), indent=1)
                for sk, objs in fin.items():
                    for n in objs:
                        names.setdefault(sk.split("|")[0], []).append(n)
                n_del = 0
                for b, ns in names.items():
                    for i in range(0, len(ns), 50):
                        native.rm_names(b, ns[i:i + 50])
                    n_del += len(ns)
                proof = {}
                for sc in native.scopes(prefix, B1, B2):
                    left = native.list_scope(*sc)
                    proof[f"gs://{sc[0]}/{sc[1]}"] = sorted(left)
                cleanup = dict(deleted=n_del, remaining=proof, empty=all(not v for v in proof.values()))
                log(f"cleanup: deleted {n_del} objects; prefix empty in both buckets = {cleanup['empty']}")
            except Exception as ex:   # noqa
                cleanup = dict(error=str(ex), empty=False)
                log(f"CLEANUP FAILED: {ex}")
        meta["cleanup"] = cleanup
        meta["elapsed_s"] = round(time.time() - t_run0, 1)
        tot = {"gcloud": 0, "n_calls": 0}
        if os.path.exists(ledger):
            for ln in open(ledger):
                e = json.loads(ln)
                tot["gcloud"] += e["n"]
                tot["n_calls"] += 1
                tot.setdefault("by_who", {})
                tot["by_who"][e["who"]] = tot["by_who"].get(e["who"], 0) + e["n"]
        meta["native_requests"] = tot
        json.dump(meta, open(os.path.join(run, "run.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
