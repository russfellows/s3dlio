python bench_s3-torch_v8.py \
  --bucket my-bucket2 \
  --prefix bench-$(date +%H:%M) \
  --mb 8 \          
  --threads 16 \
  --gbps 1000

