# Python test script to show python binding capabilities
import time
import asyncio
import s3dlio as s3

# Adjust these to your S3 URIs:
READ_PREFIX = "s3://my-bucket2/my-data2/"
TEST_PREFIX = "s3://my-bucket2/test_rust_api/"
NUM_OBJECTS = 500
TEMPLATE = "object_{}_of_{}.dat"
NUM_JOBS = 32
SIZE = 20971520  # s3.DEFAULT_OBJECT_SIZE  from Rust default


def sync_tests():

    print("=== Sync LIST ===")
    start = time.time()
    objs = s3.list(READ_PREFIX)
    dt = time.time() - start
    print(f"Found {len(objs)} objects under {READ_PREFIX} in {dt:.2f}s")


    print("\n=== Sync PUT ===")
    start = time.time()
    s3.put(TEST_PREFIX, NUM_OBJECTS, TEMPLATE, NUM_JOBS)
    dt = time.time() - start
    total_bytes = NUM_OBJECTS * SIZE
    mib = total_bytes / (1024**2)
    print(f"Uploaded {NUM_OBJECTS} objects ({mib:.2f} MiB) in {dt:.2f}s -> {NUM_OBJECTS/dt:.2f} ops/s, {mib/dt:.2f} MiB/s")
 
    print("\n=== Sync STAT ===")
    # re-build the same list of URIs we want to stat
    uris = [f"{TEST_PREFIX}" + TEMPLATE.format(i, NUM_OBJECTS) for i in range(NUM_OBJECTS)]
    stat_uri = uris[0]
    obj_meta = s3.stat(stat_uri)
    print(f"Stat for {stat_uri}:")
    for k, v in obj_meta.items():
        print(f"  {k}: {v}")

    print("\n=== Sync GET Many ===")
    uris = [f"{TEST_PREFIX}" + TEMPLATE.format(i, NUM_OBJECTS) for i in range(NUM_OBJECTS)]
    start = time.time()
    pairs = s3.get_many(uris, NUM_JOBS)
    dt = time.time() - start
    total_bytes = sum(len(b) for (_, b) in pairs)
    mib = total_bytes / (1024**2)
    print(f"Fetched {len(pairs)} objects ({mib:.2f} MiB) in {dt:.2f}s -> {len(pairs)/dt:.2f} ops/s, {mib/dt:.2f} MiB/s")

    print("\n=== Sync GET Many Stats ===")
    start = time.time()
    n, tot = s3.get_many_stats(uris, NUM_JOBS)
    dt = time.time() - start
    mib = tot / (1024**2)
    print(f"Stats: {n} objects, {tot} bytes ({mib:.2f} MiB) in {dt:.2f}s -> {n/dt:.2f} ops/s, {mib/dt:.2f} MiB/s")

    print("\n=== Sync DELETE ===")
    start = time.time()
    s3.delete(TEST_PREFIX)
    dt = time.time() - start
    print(f"Deleted test objects in {dt:.2f}s")


async def async_tests():
    # turn on info‑level logs (shows the AWS_CA_BUNDLE_PATH message)
    #s3.init_logging("info")

    print("=== Async LIST ===")
    start = time.time()
    objs = s3.list(READ_PREFIX)
    dt = time.time() - start
    print(f"Found {len(objs)} objects under {READ_PREFIX} in {dt:.2f}s")

    print("\n=== Async PUT ===")
    start = time.time()
    await s3.put_async(TEST_PREFIX, NUM_OBJECTS, TEMPLATE, NUM_JOBS)
    dt = time.time() - start
    mib = (NUM_OBJECTS * SIZE) / (1024**2)
    print(f"Uploaded {NUM_OBJECTS} objects ({mib:.2f} MiB) in {dt:.2f}s -> {NUM_OBJECTS/dt:.2f} ops/s, {mib/dt:.2f} MiB/s")

    print("\n=== Async GET Many ===")
    uris = [f"{TEST_PREFIX}" + TEMPLATE.format(i, NUM_OBJECTS) for i in range(NUM_OBJECTS)]
    start = time.time()
    pairs = await s3.get_many_async(uris, NUM_JOBS)
    dt = time.time() - start
    total_bytes = sum(len(b) for (_, b) in pairs)
    mib = total_bytes / (1024**2)
    print(f"Fetched {len(pairs)} objects ({mib:.2f} MiB) in {dt:.2f}s -> {len(pairs)/dt:.2f} ops/s, {mib/dt:.2f} MiB/s")

    print("\n=== Async GET Many Stats ===")
    start = time.time()
    n, tot = await s3.get_many_stats_async(uris, NUM_JOBS)
    dt = time.time() - start
    mib = tot / (1024**2)
    print(f"Stats: {n} objects, {tot} bytes ({mib:.2f} MiB) in {dt:.2f}s -> {n/dt:.2f} ops/s, {mib/dt:.2f} MiB/s")

    print("\n=== Async DELETE ===")
    start = time.time()
    s3.delete(TEST_PREFIX)
    dt = time.time() - start
    print(f"Deleted test objects in {dt:.2f}s")


def main():

    # turn on info‑level logs (shows the AWS S3 info messages, including AWS_CA_BUNDLE_PATH loading)
    #s3.init_logging("info")
    # turn on debug‑level logs (shows AWS S3 SDK info + debug messages)
    #s3.init_logging("debug")

    # Start op logging (warp-replay compatible TSV.ZST)
    #s3.init_op_log("/tmp/python_s3_ops_test.tsv.zst")

    sync_tests()
    print()
    asyncio.run(async_tests())

    # Flush and close (safe to call multiple times)
    #s3.finalize_op_log()



if __name__ == "__main__":
    main()

