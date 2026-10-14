"""delete_batch (Rust probe, no op-log): batch across two buckets; same-key decoys in the other bucket must survive."""
import os, sys, time, subprocess
sys.path.insert(0, os.environ["H_LIB"])
import hlib
c = hlib.Ctx()
binp = os.environ["H_DELBATCH_BIN"]
names = ("db_t1", "db_t2")      # b1/<p>/db/x1 , b2/<p>/db/x2   (decoys: b2/<p>/db/x1 , b1/<p>/db/x2)
uris = [c.uri(n) for n in names]
ents = [dict(id=f"member{i+1}", api="delete_batch(rust)", uri=u, kind="DELETE", asked=c.asked(n), expect_error=False,
             args={}, note="rust probe: no op-log in the probe", group="delete_batch", pending=True)
        for i, (n, u) in enumerate(zip(names, uris))]
c.calls.extend(ents)
c.flush()
hlib.emit_marker("BEGIN", "delete_batch")
t0 = time.time_ns()
try:
    r = subprocess.run([binp] + uris, capture_output=True, text=True, timeout=60)
    out = r.stdout.strip()[:400]
    print("PROBE rc=%s stdout=%s stderr=%s" % (r.returncode, out, r.stderr.strip()[-300:]), flush=True)
    err = None if (r.returncode == 0 and "Err(" not in r.stdout) else f"probe failed: {out} {r.stderr.strip()[-200:]}"
except BaseException as ex:  # noqa
    out, err = "", f"{type(ex).__name__}: {str(ex)[:300]}"
t1 = time.time_ns()
hlib.emit_marker("END", "delete_batch")
for e in ents:
    e.update(t_start_ns=t0, t_end_ns=t1, pending=False, outcome=dict(hang=False, ok=err is None, error=err, repr=out))
for e, (n, dn) in zip(ents, (("db_t1", "db_decoy1"), ("db_t2", "db_decoy2"))):
    e["native_after"] = c.native_check([(c.fx[n]["bucket"], c.fx[n]["key"], "target"),
                                        (c.fx[dn]["bucket"], c.fx[dn]["key"], "decoy")])
c.flush()
c.finish(oplog=False)
