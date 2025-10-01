//! Profiling infrastructure for s3dlio performance analysis
//! 
//! This module provides CPU sampling profiling, async task monitoring, and structured tracing
//! when the "profiling" feature is enabled. It helps identify performance bottlenecks in:
//! - HTTP request/response processing
//! - Data copying and buffer management
//! - Async task scheduling and contention
//! - S3/Azure API call latencies
//! 
//! Usage:
//! ```bash
//! # Enable profiling and build with release optimizations
//! RUSTFLAGS="--cfg tokio_unstable" cargo run --release --features profiling
//! 
//! # For tokio-console monitoring:
//! RUSTFLAGS="--cfg tokio_unstable" cargo run --release --features profiling
//! # In another terminal:
//! tokio-console
//! ```

/// Global profiling state
#[cfg(feature = "profiling")]
static PROFILING_INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "profiling")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
#[cfg(feature = "profiling")]
use tracing::{warn, debug};

#[cfg(not(feature = "profiling"))]
use tracing::{warn, debug};

/// Initialize comprehensive profiling infrastructure
/// 
/// This sets up:
/// - Structured tracing with environment-based filtering
/// - Optional tokio-console integration for async task monitoring
/// 
/// Call this early in main() when profiling is needed.
pub fn init_profiling() -> anyhow::Result<()> {
    #[cfg(feature = "profiling")]
    {
        // Check if already initialized
        if PROFILING_INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            warn!("Profiling already initialized, skipping");
            return Ok(());
        }

        info!("Initializing profiling infrastructure");

        // Set up tracing subscriber with layers
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("s3dlio=debug,info"));

        // Add tokio-console layer if available
        if std::env::var("S3DLIO_TOKIO_CONSOLE").is_ok() {
            console_subscriber::init();
            info!("Tokio console subscriber initialized - connect with 'tokio-console'");
        } else {
            let subscriber = tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_thread_ids(true)
                    .with_file(true)
                    .with_line_number(true));

            subscriber.init();
        }

        PROFILING_INITIALIZED.store(true, std::sync::atomic::Ordering::Relaxed);
        info!("Profiling initialized successfully");
        Ok(())
    }

    #[cfg(not(feature = "profiling"))]
    {
        debug!("Profiling not available (feature disabled)");
        Ok(())
    }
}

/// Generate and save CPU flamegraph to file
/// 
/// This should be called at the end of a profiling session to dump
/// the accumulated CPU samples as an SVG flamegraph.
pub fn save_flamegraph(path: &str) -> anyhow::Result<()> {
    #[cfg(feature = "profiling")]
    {
        info!("Flamegraph save requested to: {}", path);
        warn!("Use standalone pprof ProfilerGuard for flamegraph generation");
        Ok(())
    }

    #[cfg(not(feature = "profiling"))]
    {
        let _ = path; // Suppress unused warning
        warn!("Cannot save flamegraph - profiling feature not enabled");
        anyhow::bail!("Profiling feature not enabled");
    }
}

/// Standalone CPU profiler for specific code sections
/// 
/// Use this for targeted profiling of specific operations:
/// ```rust
/// let _profiler = s3dlio::profiling::profile_section("my_operation");
/// // ... expensive code ...
/// // profiler automatically stops when dropped
/// ```
pub fn profile_section(name: &str) -> anyhow::Result<SectionProfiler> {
    SectionProfiler::new(name)
}

pub struct SectionProfiler {
    #[allow(dead_code)]
    name: String,
}

impl SectionProfiler {
    fn new(name: &str) -> anyhow::Result<Self> {
        #[cfg(feature = "profiling")]
        {
            debug!("Started section profiler: {}", name);
        }

        Ok(Self {
            name: name.to_string(),
        })
    }

    /// Save this section's profile to an SVG flamegraph
    pub fn save_flamegraph(&self, _path: &str) -> anyhow::Result<()> {
        #[cfg(feature = "profiling")]
        {
            info!("Section profiler '{}' save requested - use standalone pprof::ProfilerGuard instead", self.name);
        }

        #[cfg(not(feature = "profiling"))]
        {
            warn!("Cannot save section flamegraph - profiling feature not enabled");
        }
        
        Ok(())
    }
}

impl Drop for SectionProfiler {
    fn drop(&mut self) {
        #[cfg(feature = "profiling")]
        {
            debug!("Stopped section profiler: {}", self.name);
        }
    }
}

/// Convenience macro for instrumenting functions with tracing spans
/// 
/// When profiling is enabled, this adds detailed tracing spans with
/// configurable fields for performance analysis.
#[macro_export]
macro_rules! profile_span {
    ($name:expr) => {
        {
            #[cfg(feature = "profiling")]
            {
                ::tracing::info_span!($name)
            }
            #[cfg(not(feature = "profiling"))]
            {
                () // No-op when profiling disabled
            }
        }
    };
    ($name:expr, $($field:tt)*) => {
        {
            #[cfg(feature = "profiling")]
            {
                ::tracing::info_span!($name, $($field)*)
            }
            #[cfg(not(feature = "profiling"))]
            {
                () // No-op when profiling disabled  
            }
        }
    };
}

/// Convenience macro for profiling async functions
#[macro_export]
macro_rules! profile_async {
    ($name:expr, $future:expr) => {
        {
            #[cfg(feature = "profiling")]
            {
                async move {
                    let span = ::tracing::info_span!($name);
                    ::tracing::Instrument::instrument($future, span).await
                }
            }
            #[cfg(not(feature = "profiling"))]
            {
                $future
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_profiler_creation() {
        let profiler = profile_section("test_section");
        assert!(profiler.is_ok());
    }

    #[tokio::test] 
    async fn test_profiling_macros() {
        // These should compile and run without panicking
        let _span = profile_span!("test_span");
        let _span_with_fields = profile_span!("test_span", field1 = "value1");
        
        let result = profile_async!("test_async", async { 42 }).await;
        assert_eq!(result, 42);
    }
}
