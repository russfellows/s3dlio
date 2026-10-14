"""multi_endpoint: Python create_multi_endpoint_store([s3://<bucket1>/]) -> get / get_range / put / list / delete."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()

c.oplog_init()      # BEFORE the store is created (stores capture the logger at construction)
me = s3dlio.create_multi_endpoint_store([f"s3://{c.b1}/"], "round_robin")
um = c.uri("me_m1")
cm = c.content("me_m1")
def run(coro_fn):
    return asyncio.run(coro_fn())
async def _get():
    return await me.get(um)
c.call("get", "me.get", um, lambda: run(_get), kind="GET", asked=c.asked("me_m1"), expect_bytes=cm)
async def _rng():
    return await me.get_range(um, 10, 40)
c.call("get_range", "me.get_range", um, lambda: run(_rng), kind="GET_RANGE", asked=c.asked("me_m1"),
       args=dict(offset=10, length=40), expect_bytes=cm[10:50])
b, k = c.new_key("me/p1")
pu = f"s3://{b}/{k}"
pl = (f"ME-PUT-{c.seed}-" * 20).encode()[:110]
async def _put():
    return await me.put(pu, pl)
c.call("put", "me.put", pu, lambda: run(_put), kind="PUT",
       asked=c.asked_new(b, k), payload=pl, watch=[(b, k, "target")])
dp = f"s3://{c.b1}/{c.prefix}/me/"
async def _list():
    return await me.list(dp, False)
c.call("list", "me.list", dp, lambda: run(_list), kind="LIST", asked=c.asked_new(c.b1, c.prefix + "/me/"))
async def _del():
    return await me.delete(pu)
c.call("delete", "me.delete", pu, lambda: run(_del), kind="DELETE", asked=c.asked_new(b, k), watch=[(b, k, "target")])
c.finish()
