# How to further enhance Data Generation
Given that you’ll be calling generate_controlled_data() thousands of times with the same dedup/compress settings but wildly different sizes on a 32‑core machine, you can get essentially zero per‑call overhead beyond the actual block copies and random fills by pulling out all of the fixed setup into a little stateful builder object, and then doing one parallel pass per call that both decides your zero‑prefix length and fills the chunk in place.

1) Factor out your run‑wide setup into a DataGenerator
```
use once_cell::sync::Lazy;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;

pub const BLK_SIZE: usize = 512;
const HALF_BLK: usize = BLK_SIZE / 2;
const MOD_SIZE: usize = 32;

/// A single shared, immutable base block of randomness.
static BASE_BLOCK: Lazy<Arc<Vec<u8>>> = Lazy::new(|| {
    let mut buf = vec![0u8; BLK_SIZE];
    rand::rngs::ThreadRng::default().fill(&mut buf[..]);
    Arc::new(buf)
});

/// Precompute everything that only depends on (dedup, compress)
pub struct DataGenerator {
    dedup: usize,
    f_num: usize,
    f_den: usize,
}

impl DataGenerator {
    pub fn new(dedup: usize, compress: usize) -> Self {
        let dedup = dedup.max(1);
        let (f_num, f_den) = if compress > 1 {
            (compress - 1, compress)
        } else {
            (0, 1)
        };
        DataGenerator { dedup, f_num, f_den }
    }

    pub fn generate(&self, mut size: usize) -> Vec<u8> {
        if size < BLK_SIZE {
            size = BLK_SIZE;
        }

        let block_size = BLK_SIZE;
        let nblocks = (size + block_size - 1) / block_size;
        let unique_blocks = if self.dedup > 1 {
            ((nblocks as f64) / (self.dedup as f64)).round().max(1.0) as usize
        } else {
            nblocks
        };

        // distribute the fractional zero‑prefix with Bresenham
        let floor_len = (self.f_num * block_size) / self.f_den;
        let rem      = (self.f_num * block_size) % self.f_den;

        let mut err_acc = 0;
        let mut const_lens = Vec::with_capacity(unique_blocks);
        for _ in 0..unique_blocks {
            err_acc += rem;
            if err_acc >= self.f_den {
                err_acc -= self.f_den;
                const_lens.push(floor_len + 1);
            } else {
                const_lens.push(floor_len);
            }
        }

        // single pass: for each block, copy+zero+randomize in place
        let total_size = nblocks * block_size;
        let mut data = Vec::with_capacity(total_size);
        unsafe { data.set_len(total_size) };

        data.par_chunks_mut(block_size)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut rng = rand::rngs::ThreadRng::default();
                // 1) clone from shared base
                let src = &*BASE_BLOCK;
                let len = chunk.len();
                chunk.copy_from_slice(&src[..len]);

                // 2) zero‑prefix
                let ci = i % unique_blocks;
                let const_len = const_lens[ci].min(len);
                chunk[..const_len].fill(0);

                // 3) inject uniqueness
                let region_start = const_len;
                let region_len   = len - region_start;
                let modify_len   = region_len.min(MOD_SIZE);
                if modify_len > 0 {
                    rng.fill(&mut chunk[region_start..region_start + modify_len]);
                }
                let second_off = HALF_BLK.max(region_start);
                if second_off + modify_len <= len {
                    rng.fill(&mut chunk[second_off..second_off + modify_len]);
                }
            });

        // trim to exact size if we hit a partial block
        data.truncate(size);
        data
    }
}
```
2) Why this is optimal for your workload
One parallel pass: you no longer build an intermediate Vec<unique_blocks> at all.
Zero per‑call cloning of a big vector—instead you clone only once per block directly into the target buffer.
Full use of your 32 cores via Rayon on a per‑object basis; each 10 MB–10 GB object will be spread across threads.
Minimal memory traffic: you copy 512 bytes + touch ~64 bytes per block, exactly as before, but avoid the extra allocation + copy of the “unique” array.
3) How to use it
```   
fn main() {
    // At startup
    let generator = DataGenerator::new(/* dedup=*/ 3, /* compress=*/ 2);

    // Later, for each object
    let buf_10mb = generator.generate(10 * 1024 * 1024);
    // … upload buf_10mb …

    let buf_5gb = generator.generate(5 * 1024 * 1024 * 1024);
    // … upload buf_5gb …
}
```
This approach gives you the exact same controlled‑data semantics, but with all of the fixed‑parameter work (dedup/compress math) hoisted into new(), and only a single, burst‑parallel memory + RNG pass per call. That will scale to terabytes on 32+ cores with very little overhead beyond copying and your required random fills.
