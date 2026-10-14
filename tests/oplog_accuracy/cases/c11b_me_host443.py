"""me_host443: MultiEndpointStore over a host-form :443 endpoint, get (known: dialed over http:// -> fails; intermittent SIGSEGV at exit)."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()
f = c.fx["me_m1"]
hf_host = os.environ["H_HOSTFORM_HOST"]
u = f"s3://{hf_host}:443/{f['bucket']}/{f['key']}"
me = s3dlio.create_multi_endpoint_store([f"s3://{hf_host}:443/" + f["bucket"] + "/"], "round_robin")
async def _get():
    return await me.get(u)
c.call("c1", "me.get", u, lambda: asyncio.run(_get()), kind="GET",
       asked=dict(server=f"https://{hf_host}:443", bucket=f["bucket"], key=f["key"], hostform=True),
       expect_bytes=c.content("me_m1"))
c.finish()
