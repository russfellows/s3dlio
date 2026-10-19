# pytorch_smoke.py
# s3.init_logging("info")
import inspect
import os
import sys

from s3dlio import list as list_uri
from s3dlio import list_full_uris, stat
from s3dlio.torch import S3IterableDataset
from torch.utils.data import DataLoader

print(inspect.signature(S3IterableDataset.from_prefix))

# Manual smoke script (not a pytest test): reads from a live bucket you name explicitly.
# S3DLIO_TEST_BUCKET must contain a "health/" prefix with some objects; no default bucket.
BUCKET = os.environ.get("S3DLIO_TEST_BUCKET", "").strip()
if not BUCKET:
    sys.exit(
        "S3DLIO_TEST_BUCKET is not set; set it to a bucket you own to run this smoke script."
    )
URI = f"s3://{BUCKET}/health/"

uris = list_full_uris(URI)
keys = list_uri(URI)
print("object count:", len(keys))
print("first 3 keys:", keys[:3])

ds = S3IterableDataset.from_prefix(URI, prefetch=8)  # ← no enable_sharding
loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=lambda xs: xs)

for i, batch in enumerate(loader):
    # batch is a list[Tensor(uint8)] (variable-length), so compare lengths:
    sizes = [int(t.numel()) for t in batch]
    base = i * 2
    s3_sizes = [stat(uris[base + j])["size"] for j in range(len(batch))]
    print(f"batch {i} tensor-bytes={sizes}  s3-sizes={s3_sizes}")
    assert sizes == s3_sizes
    if i == 1:
        break
