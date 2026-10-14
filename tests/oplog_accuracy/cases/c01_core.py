"""core: put/get/get_range/stat/list(non-recursive, recursive)/delete + a missing-object get/stat, on bucket 1."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()

payload = (f"CORE-PUT-{c.seed}-" * 20).encode()[:100]
b, k = c.new_key("core/p1")
u = f"s3://{b}/{k}"
c.call("put", "put_bytes", u, lambda: s3dlio.put_bytes(u, payload), kind="PUT",
       asked=c.asked_new(b, k), payload=payload, watch=[(b, k, "target")])
ua = c.uri("data_a")
a = c.content("data_a")
c.call("get", "get", ua, lambda: s3dlio.get(ua), kind="GET", asked=c.asked("data_a"), expect_bytes=a)
c.call("get_range", "get_range", ua, lambda: s3dlio.get_range(ua, 10, 40), kind="GET_RANGE",
       asked=c.asked("data_a"), args=dict(offset=10, length=40), expect_bytes=a[10:50])
c.call("stat", "stat", ua, lambda: s3dlio.stat(ua), kind="STAT", asked=c.asked("data_a"))
d = f"s3://{c.b1}/{c.prefix}/data/"
c.call("list", "list", d, lambda: s3dlio.list(d), kind="LIST", asked=c.asked_new(c.b1, c.prefix + "/data/"),
       args=dict(recursive=False))
c.call("list_rec", "list", d, lambda: s3dlio.list(d, recursive=True), kind="LIST",
       asked=c.asked_new(c.b1, c.prefix + "/data/"), args=dict(recursive=True))
c.call("delete", "delete", u, lambda: s3dlio.delete(u), kind="DELETE", asked=c.asked_new(b, k), watch=[(b, k, "target")])
bm, km = c.new_key("core/nope")
um = f"s3://{bm}/{km}"
c.call("get_missing", "get", um, lambda: s3dlio.get(um), kind="GET", asked=c.asked_new(bm, km), expect_error=True)
c.call("stat_missing", "stat", um, lambda: s3dlio.stat(um), kind="STAT", asked=c.asked_new(bm, km), expect_error=True)
c.call("exists", "exists", ua, lambda: s3dlio.exists(ua), kind="STAT", asked=c.asked("data_a"), args=dict(exists=True))
c.call("exists_missing", "exists", um, lambda: s3dlio.exists(um), kind="STAT", asked=c.asked_new(bm, km),
       args=dict(exists=False))
c.finish()
