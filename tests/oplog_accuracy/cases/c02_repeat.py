"""repeat: the SAME object accessed 3 times by get and 3 times by stat."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()
c.oplog_init()

ua = c.uri("data_a")
a = c.content("data_a")
for i in range(3):
    c.call(f"get{i}", "get", ua, lambda: s3dlio.get(ua), kind="GET", asked=c.asked("data_a"), expect_bytes=a)
for i in range(3):
    c.call(f"stat{i}", "stat", ua, lambda: s3dlio.stat(ua), kind="STAT", asked=c.asked("data_a"))
c.finish()
