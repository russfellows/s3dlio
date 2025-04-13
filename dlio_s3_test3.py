import time
import dlio_s3_rust as s3

NumJobs = 32;

# List the objects (for reference)
print(s3.list("s3://my-bucket2/my-data2/")[:25])

# Test a single object download speed (optional)
start_single = time.perf_counter()
data = s3.get("s3://my-bucket2/my-data2/file_101_of_1000.npz")
end_single = time.perf_counter()
elapsed_single = end_single - start_single
mb_single = len(data) / (1024 * 1024)
speed_single = mb_single / elapsed_single
print(f"Single download: {speed_single:.2f} MB/s ({len(data)} bytes in {elapsed_single:.2f} s)")

print("")

# Measure multi-object download speed
uris = [f"s3://my-bucket2/my-data2/file_{i}_of_1000.npz" for i in range(10, 500)]
start_multi = time.perf_counter()
pairs = s3.get_many(uris, NumJobs)  # Limit to NumJobs concurrent downloads.
end_multi = time.perf_counter()

elapsed_multi = end_multi - start_multi
num_objects = len(pairs)
objects_per_second = num_objects / elapsed_multi

total_bytes = sum(len(data) for _, data in pairs)
mb_total = total_bytes / (1024 * 1024)
speed_multi = mb_total / elapsed_multi

print(f"Downloaded {num_objects} objects in {elapsed_multi:.2f} seconds")
print(f"Throughput: {objects_per_second:.2f} objects/s, {speed_multi:.2f} MB/s")

# Optionally, list each object with its size:
#for name, data in pairs:
    #print(f"{name}: {len(data)} bytes")

