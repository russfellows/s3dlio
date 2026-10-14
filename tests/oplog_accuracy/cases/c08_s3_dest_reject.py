"""s3_dest_reject: init_op_log with an s3:// destination must be rejected and create nothing; a corrected init then works."""
import os, sys, asyncio
sys.path.insert(0, os.environ["H_LIB"])
import hlib
import s3dlio
s3dlio.init_logging("debug")
c = hlib.Ctx()

c.flush()                                     # ground.json exists from here on
before = sorted(os.listdir(c.case_dir))
dest = f"s3://{c.b1}/{c.prefix}/oplog/ops.tsv.zst"
c.oplog_init(path=dest, form="path")          # init_op_log("s3://...")  -> must raise
bad = c.inits[-1]
bad["active_after"] = s3dlio.is_op_log_active()
bad["local_listing_after"] = sorted(os.listdir(c.case_dir))
bad["local_listing_before"] = before
bad["s3_dir_created"] = os.path.exists(os.path.join(c.case_dir, "s3:"))
bad["native_dest"] = c.native_check([(c.b1, c.prefix + "/oplog/ops.tsv.zst", "target")])
c.flush()
c.oplog_init()                                # the corrected (local) init must succeed
ua = c.uri("data_a")
c.call("get_after_reinit", "get", ua, lambda: s3dlio.get(ua), kind="GET", asked=c.asked("data_a"),
       expect_bytes=c.content("data_a"))
c.finish()
