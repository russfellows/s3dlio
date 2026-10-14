"""get_many: one call, three objects."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()

names = ["many_0", "many_1", "many_2"]
uris = [c.uri(n) for n in names]
items = [dict(id=f"many{i}", uri=c.uri(n), kind="GET", asked=c.asked(n), expect_bytes=c.content(n))
         for i, n in enumerate(names)]
c.call_group("get_many", "get_many", lambda: s3dlio.get_many(uris), items)
c.finish()
