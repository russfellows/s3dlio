# pytorch_smoke.py
from s3dlio.torch import S3IterableDataset
from s3dlio import list as list_uri, stat
from torch.utils.data import DataLoader
import s3dlio as s3
from s3dlio import list_uris, stat


#s3.init_logging("info")

import inspect
from s3dlio.torch import S3IterableDataset
print(inspect.signature(S3IterableDataset.from_prefix))

URI = "s3://russ-b3/health/"

uris = list_uris(URI)
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

