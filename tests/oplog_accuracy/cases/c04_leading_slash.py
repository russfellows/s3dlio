"""leading_slash: key with a leading slash (s3://bucket//<prefix>/x) next to the plain key."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()

for nm in ("slash_lead", "slash_plain"):
    u = c.uri(nm)
    cc = c.content(nm)
    c.call(f"get_{nm}", "get", u, lambda u=u: s3dlio.get(u), kind="GET", asked=c.asked(nm), expect_bytes=cc)
for nm in ("slash_lead", "slash_plain"):
    u = c.uri(nm)
    c.call(f"stat_{nm}", "stat", u, lambda u=u: s3dlio.stat(u), kind="STAT", asked=c.asked(nm))
for nm in ("slash_lead", "slash_plain"):
    u = c.uri(nm)
    cc = c.content(nm)
    c.call(f"range_{nm}", "get_range", u, lambda u=u: s3dlio.get_range(u, 5, 20), kind="GET_RANGE",
           asked=c.asked(nm), args=dict(offset=5, length=20), expect_bytes=cc[5:25])
b, k = c.new_key("slash/pnew", lead=True)
pl = (f"LEAD-PUT-{c.seed}-" * 20).encode()[:140]
u = f"s3://{b}/{k}"          # -> s3://<bucket1>//<prefix>/slash/pnew
c.call("put_lead", "put_bytes", u, lambda: s3dlio.put_bytes(u, pl), kind="PUT",
       asked=c.asked_new(b, k), payload=pl,
       watch=[(b, k, "target"), (b, k.lstrip("/"), "alt")])
u = c.uri("dls_lead")        # delete the leading-slash object; the plain one must survive
c.call("delete_lead", "delete", u, lambda: s3dlio.delete(u), kind="DELETE", asked=c.asked("dls_lead"),
       watch=[(c.b1, c.fx["dls_lead"]["key"], "target"), (c.b1, c.fx["dls_plain"]["key"], "decoy")])
c.finish()
