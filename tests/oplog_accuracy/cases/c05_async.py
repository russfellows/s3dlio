"""async: put_bytes_async, get_async (twice), stat_async, then a sync delete of the async-written object."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()

def run(coro_fn, *a):
    return asyncio.run(coro_fn(*a))

b, k = c.new_key("as/p1")
pl = (f"ASYNC-PUT-{c.seed}-" * 20).encode()[:150]
u = f"s3://{b}/{k}"
async def _put():
    return await s3dlio.put_bytes_async(u, pl)
c.call("put_async", "put_bytes_async", u, lambda: run(_put), kind="PUT",
       asked=c.asked_new(b, k), payload=pl, watch=[(b, k, "target")])
ug = c.uri("async_g1")
async def _get():
    return await s3dlio.get_async(ug)
for i in range(2):
    c.call(f"get_async{i}", "get_async", ug, lambda: run(_get), kind="GET", asked=c.asked("async_g1"),
           expect_bytes=c.content("async_g1"))
async def _stat():
    return await s3dlio.stat_async(ug)
c.call("stat_async", "stat_async", ug, lambda: run(_stat), kind="STAT", asked=c.asked("async_g1"))
c.call("delete", "delete", u, lambda: s3dlio.delete(u), kind="DELETE", asked=c.asked_new(b, k), watch=[(b, k, "target")])
c.finish()
