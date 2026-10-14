"""hostform: host-form URIs  s3://<host>[:port]/<bucket>/<key>.
H_MODE=host443    s3://<H_HOSTFORM_HOST>:443/...  (a real TLS host serving the buckets; known: plain http sent to 443 on some paths; sync calls may hang)
H_MODE=portless   s3://10.255.255.1/...                 (known: host ignored, call goes to the configured endpoint)
H_OP=get|stat|get_range|get_async|get_many|put
The case env may point AWS_ENDPOINT_URL at an unroutable DECOY, so any call routed to the configured endpoint
instead of the URI's host fails visibly (and shows in the client log) instead of silently succeeding."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()
mode, op = os.environ["H_MODE"], os.environ["H_OP"]
if mode == "host443":
    hf_host = os.environ["H_HOSTFORM_HOST"]
    host, server, expect_error = f"{hf_host}:443", f"https://{hf_host}:443", False
else:
    host, server, expect_error = "10.255.255.1", "10.255.255.1", True      # nothing answers there: a correct client fails
def hf(name):
    f = c.fx[name]
    return f"s3://{host}/{f['bucket']}/{f['key']}", dict(server=server, bucket=f["bucket"], key=f["key"], hostform=True)
u, asked = hf("data_b" if mode == "host443" else "data_c")
cb = c.content("data_b" if mode == "host443" else "data_c")
if op == "get":
    c.call("c1", "get", u, lambda: s3dlio.get(u), kind="GET", asked=asked, expect_bytes=cb, expect_error=expect_error)
elif op == "stat":
    c.call("c1", "stat", u, lambda: s3dlio.stat(u), kind="STAT", asked=asked, expect_error=expect_error)
elif op == "get_range":
    c.call("c1", "get_range", u, lambda: s3dlio.get_range(u, 10, 40), kind="GET_RANGE", asked=asked,
           args=dict(offset=10, length=40), expect_bytes=cb[10:50], expect_error=expect_error)
elif op == "get_async":
    async def _g():
        return await s3dlio.get_async(u)
    c.call("c1", "get_async", u, lambda: asyncio.run(_g()), kind="GET", asked=asked, expect_bytes=cb,
           expect_error=expect_error)
elif op == "get_many":
    u2, asked2 = hf("data_a")
    items = [dict(id="m1", uri=u, kind="GET", asked=asked, expect_bytes=cb),
             dict(id="m2", uri=u2, kind="GET", asked=asked2, expect_bytes=c.content("data_a"))]
    c.call_group("get_many", "get_many", lambda: s3dlio.get_many([u, u2]), items)
elif op == "put":
    k = c.prefix + "/pl/put-np"
    u = f"s3://{host}/{c.b1}/{k}"
    pl = (f"HOSTFORM-PUT-{c.seed}-" * 20).encode()[:160]
    c.call("c1", "put_bytes", u, lambda: s3dlio.put_bytes(u, pl), kind="PUT",
           asked=dict(server=server, bucket=c.b1, key=k, hostform=True), payload=pl,
           expect_error=expect_error, watch=[(c.b1, k, "target")])
c.finish()
