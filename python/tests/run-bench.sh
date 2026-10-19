python bench_s3-torch_v8.py \
  --bucket "${S3DLIO_TEST_BUCKET:?set S3DLIO_TEST_BUCKET to an S3 bucket you own}" \
  --prefix bench-$(date +%H:%M) \
  --mb 8 \
  --threads 16 \
  --gbps 1000

