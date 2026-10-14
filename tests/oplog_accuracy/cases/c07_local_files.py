"""local_files: file:// put/get/get_range/stat/list/delete in the SAME directory as the op-log (the op-log must never record itself)."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()

# the op-log was created by oplog_init() in c.case_dir; the files live next to it
d = c.case_dir
f1 = os.path.join(d, "f1.bin")
payload = (f"LOCAL-{c.seed}-" * 40).encode()[:200]
u = "file://" + f1
asked = dict(server=None, bucket=None, key=f1, scheme="file")
c.call("put", "put_bytes", u, lambda: s3dlio.put_bytes(u, payload), kind="PUT",
       asked=asked, payload=payload)
c.call("get", "get", u, lambda: s3dlio.get(u), kind="GET", asked=asked, expect_bytes=payload)
c.call("get_range", "get_range", u, lambda: s3dlio.get_range(u, 7, 30), kind="GET_RANGE", asked=asked,
       args=dict(offset=7, length=30), expect_bytes=payload[7:37])
c.call("stat", "stat", u, lambda: s3dlio.stat(u), kind="STAT", asked=asked)
dd = "file://" + d + "/"
c.call("list", "list", dd, lambda: s3dlio.list(dd), kind="LIST", asked=dict(asked, key=d + "/"))
c.call("delete", "delete", u, lambda: s3dlio.delete(u), kind="DELETE", asked=asked)
um = "file://" + os.path.join(d, "nope.bin")
c.call("get_missing", "get", um, lambda: s3dlio.get(um), kind="GET", asked=dict(asked, key=os.path.join(d, "nope.bin")),
       expect_error=True)

c.finish()
