"""two_buckets: identical key names in bucket 1 and bucket 2 (get, stat, put, delete)."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()

for nm in ("same_b1", "same_b2"):
    u = c.uri(nm)
    c.call(f"get_{nm}", "get", u, lambda u=u: s3dlio.get(u), kind="GET", asked=c.asked(nm), expect_bytes=c.content(nm))
for nm in ("same_b1", "same_b2"):
    u = c.uri(nm)
    c.call(f"stat_{nm}", "stat", u, lambda u=u: s3dlio.stat(u), kind="STAT", asked=c.asked(nm))
rel = "tb/put"
for i, bk in enumerate((c.b1, c.b2)):
    b, k = c.new_key(rel, bucket=bk)
    pl = (f"TWOB-{i}-{c.seed}-" * 20).encode()[:120 + i]
    u = f"s3://{b}/{k}"
    c.call(f"put_{bk}", "put_bytes", u, lambda u=u, pl=pl: s3dlio.put_bytes(u, pl), kind="PUT",
           asked=c.asked_new(b, k), payload=pl,
           watch=[(b, k, "target"), (c.b2 if bk == c.b1 else c.b1, k, "alt")])
# delete the key that exists in BOTH buckets, from bucket 2 only (decoy in bucket 1 must survive)
u = c.uri("del_a_b2")
c.call("delete_b2", "delete", u, lambda: s3dlio.delete(u), kind="DELETE", asked=c.asked("del_a_b2"),
       watch=[(c.b2, c.fx["del_a_b2"]["key"], "target"), (c.b1, c.fx["del_a_b1"]["key"], "decoy")])
c.finish()
