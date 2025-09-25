use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use std::sync::Arc;
use std::time::Duration;

/// Creates a progress bar for S3 operations with warp-style formatting
pub struct S3ProgressTracker {
    pub multi: MultiProgress,
    pub progress_bar: ProgressBar,
}

impl S3ProgressTracker {
    /// Create a new progress tracker for S3 operations
    pub fn new(operation: &str, total_objects: u64, total_bytes: u64) -> Self {
        let multi = MultiProgress::new();
        let pb = multi.add(ProgressBar::new(total_bytes));
        
        // Warp-style progress bar template
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    &format!(
                        "{}: {{spinner:.green}} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{bytes}}/{{total_bytes}} ({{bytes_per_sec}}, ETA: {{eta}})",
                        operation
                    )
                )
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  ")
        );

        // Add a message showing object progress
        pb.set_message(format!("0/{} objects", total_objects));

        Self {
            multi,
            progress_bar: pb,
        }
    }

    /// Update progress with bytes transferred and objects completed
    pub fn update(&self, bytes_transferred: u64, objects_completed: u64, total_objects: u64) {
        self.progress_bar.set_position(bytes_transferred);
        self.progress_bar.set_message(format!("{}/{} objects", objects_completed, total_objects));
    }

    /// Update the total size of the progress bar (useful when total size is discovered during operation)
    pub fn set_total_bytes(&self, total_bytes: u64) {
        self.progress_bar.set_length(total_bytes);
    }

    /// Finish the progress bar with a completion message
    pub fn finish(&self, operation: &str, total_bytes: u64, duration: Duration) {
        let throughput_mbps = (total_bytes as f64 / 1_048_576.0) / duration.as_secs_f64();
        
        self.progress_bar.finish_with_message(
            format!(
                "{} complete! {:.2} MB in {:.2}s ({:.2} MB/s)",
                operation,
                total_bytes as f64 / 1_048_576.0,
                duration.as_secs_f64(),
                throughput_mbps
            )
        );
    }

    /// Create a simple spinner for operations without known total size
    pub fn spinner(operation: &str) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template(&format!("{}: {{spinner:.green}} {{msg}}", operation))
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        );
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    }
}

/// Simple progress callback for use with s3dlio functions
pub struct ProgressCallback {
    pub tracker: Arc<S3ProgressTracker>,
    pub objects_completed: std::sync::atomic::AtomicU64,
    pub bytes_transferred: std::sync::atomic::AtomicU64,
    pub total_objects: u64,
}

impl ProgressCallback {
    pub fn new(tracker: Arc<S3ProgressTracker>, total_objects: u64) -> Self {
        Self {
            tracker,
            objects_completed: std::sync::atomic::AtomicU64::new(0),
            bytes_transferred: std::sync::atomic::AtomicU64::new(0),
            total_objects,
        }
    }

    /// Call this when an object transfer completes
    pub fn object_completed(&self, bytes: u64) {
        let completed = self.objects_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let total_bytes = self.bytes_transferred.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed) + bytes;
        
        self.tracker.update(total_bytes, completed, self.total_objects);
    }

    /// Update the progress bar's total size as we discover the actual total
    pub fn update_total_bytes(&self, total_bytes: u64) {
        self.tracker.set_total_bytes(total_bytes);
    }
}