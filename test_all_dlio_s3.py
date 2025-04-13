import time
import dlio_s3_rust as s3

def main():
    NumJobs = 32
    # Define the prefixes (ensure that these buckets exist or can be created):
    read_prefix = "s3://my-bucket2/my-data2/"
    test_prefix = "s3://my-bucket2/test_rust_api/"

    # --- 1. List objects in an existing prefix ---
    print("=== Listing objects from", read_prefix, "===")
    initial_list = s3.list(read_prefix)
    print("First 25 objects:")
    print(initial_list[:25])
    print("Total objects:", len(initial_list))
    print("")

    # --- 2. Single object download test ---
    test_get_uri = "s3://my-bucket2/my-data2/file_101_of_1000.npz"
    start_single = time.perf_counter()
    data = s3.get(test_get_uri)
    end_single = time.perf_counter()
    elapsed_single = end_single - start_single
    mb_single = len(data) / (1024 * 1024)
    speed_single = mb_single / elapsed_single
    print(f"Single download from {test_get_uri}: {speed_single:.2f} MB/s ({len(data)} bytes in {elapsed_single:.2f} s)")
    print("")

    # --- 3. Multi-object download test ---
    uris = [f"s3://my-bucket2/my-data2/file_{i}_of_1000.npz" for i in range(10, 500)]
    start_multi = time.perf_counter()
    pairs = s3.get_many(uris, NumJobs)
    end_multi = time.perf_counter()
    elapsed_multi = end_multi - start_multi
    num_objects = len(pairs)
    total_bytes = sum(len(d) for _, d in pairs)
    mb_total = total_bytes / (1024 * 1024)
    speed_multi = mb_total / elapsed_multi
    objects_per_second = num_objects / elapsed_multi
    print(f"Multi-get: Downloaded {num_objects} objects in {elapsed_multi:.2f} s")
    print(f"Throughput: {objects_per_second:.2f} objects/s, {speed_multi:.2f} MB/s")
    print("")

    # --- 4. Single object upload (put) test ---
    # For a single object put, we use a prefix that will generate one object.
    # Note: since our Python API generates the object name from the template, if we set a template
    # that does not include '{}' the template is taken literally.
    single_put_prefix = test_prefix + "single_put/"
    put_size = 1 * 1024 * 1024  # 1 MB
    start_put = time.perf_counter()
    # Call put with num=1; template is set to the fixed name "single_test.dat"
    s3.put(single_put_prefix, num=1, template="single_test.dat", max_in_flight=NumJobs, size=put_size, should_create_bucket=False)
    end_put = time.perf_counter()
    elapsed_put = end_put - start_put
    speed_put = (put_size / (1024 * 1024)) / elapsed_put
    print(f"Single put: Uploaded 1 MB to {single_put_prefix} as 'single_test.dat' in {elapsed_put:.2f} s ({speed_put:.2f} MB/s)")
    print("")

    # --- 5. Download the single uploaded object ---
    # The generated object’s full URI will be the prefix with the template name appended.
    single_put_object = single_put_prefix + "single_test.dat"
    start_get_new = time.perf_counter()
    new_data = s3.get(single_put_object)
    end_get_new = time.perf_counter()
    elapsed_get_new = end_get_new - start_get_new
    print(f"Get single put: Retrieved {len(new_data)} bytes from {single_put_object} in {elapsed_get_new:.2f} s")
    print("")

    # --- 6. Multiple object upload (put) test ---
    # For multiple objects, we set a template with '{}' in it so that each object gets a distinct name.
    multi_put_prefix = test_prefix + "multi_put/"
    num_put_many = 10
    put_many_size = 1 * 1024 * 1024  # 1 MB each
    start_put_many = time.perf_counter()
    s3.put(multi_put_prefix, num=num_put_many, template="multi_test_{}.dat", max_in_flight=NumJobs, size=put_many_size, should_create_bucket=False)
    end_put_many = time.perf_counter()
    elapsed_put_many = end_put_many - start_put_many
    total_put_bytes = num_put_many * put_many_size
    speed_put_many = (total_put_bytes / (1024 * 1024)) / elapsed_put_many
    print(f"Multi put: Uploaded {num_put_many} objects (total {total_put_bytes} bytes) to {multi_put_prefix} in {elapsed_put_many:.2f} s ({speed_put_many:.2f} MB/s)")
    print("")

    # --- 7. Multiple object download (get_many) test for the multi-put objects ---
    # Reconstruct the list of expected URIs based on the same prefix/template logic.
    multi_put_uris = [multi_put_prefix + f"multi_test_{i}.dat" for i in range(num_put_many)]
    start_get_many_new = time.perf_counter()
    put_many_pairs = s3.get_many(multi_put_uris, NumJobs)
    end_get_many_new = time.perf_counter()
    elapsed_get_many_new = end_get_many_new - start_get_many_new
    total_get_many_bytes = sum(len(d) for _, d in put_many_pairs)
    mb_get_many = total_get_many_bytes / (1024 * 1024)
    speed_get_many = mb_get_many / elapsed_get_many_new
    print(f"Multi get: Retrieved {len(put_many_pairs)} objects (total {total_get_many_bytes} bytes) in {elapsed_get_many_new:.2f} s ({speed_get_many:.2f} MB/s)")
    print("")

    # --- 8. Delete test: Remove all objects under the test prefix ---
    start_delete = time.perf_counter()
    s3.delete(test_prefix)
    end_delete = time.perf_counter()
    elapsed_delete = end_delete - start_delete
    print(f"Delete: Removed all objects under {test_prefix} in {elapsed_delete:.2f} s")
    print("")

    # --- 9. Verify deletion ---
    post_delete_list = s3.list(test_prefix)
    print(f"After deletion, objects under {test_prefix}:")
    print(post_delete_list)
    if not post_delete_list:
        print("Deletion verified: no objects remain.")
    else:
        print("Warning: Some objects still remain!")

if __name__ == "__main__":
    main()

