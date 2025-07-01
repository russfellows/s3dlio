# Efficiently Logging in Rust
To efficiently log from a tight async loop without re-opening the file, you should use a dedicated logging crate that supports non-blocking, buffered I/O. The standard modern approach in Rust is using the tracing ecosystem, specifically with the tracing-appender crate.

This strategy works by moving the actual file writing to a separate, dedicated background thread. Your fast async tasks simply send the log messages to an in-memory channel, which is a very quick, non-blocking operation. The dedicated thread then reads from this channel and writes the data to the log file in efficient batches.

## Step-by-Step Implementation 🦀

Here’s how to set this up.

1. Add Dependencies
Add the following to your Cargo.toml:

```
[dependencies]
tokio = { version = "1", features = ["full"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-appender = "0.2"
```

2\. Configure the Non-Blocking Writer

In your main.rs, you'll set up the logger once when the application starts. The _guard variable is important—it must be kept in scope for the entire duration of your program to ensure logs are flushed when the program exits.

Rust

```
use std::time::Duration;
use tracing::info;
use tracing_appender::rolling;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    // 1. Set up a non-blocking rolling file appender.
    // This will create a new log file every day.
    let file_appender = rolling::daily("./", "app.log");
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(file_appender);

    // 2. Configure the tracing subscriber.
    // This directs logs to both the console (for debugging) and the file.
    tracing_subscriber::registry()
        .with(fmt::Layer::new().with_writer(std::io::stdout))
        .with(fmt::Layer::new().with_writer(non_blocking_appender))
        .init();

    // This guard must be kept alive for the program's duration.
    // If it's dropped, the background logging thread will exit.

    info!("Logger initialized. Starting async operations.");

    // 3. Run your tight async loop.
    let mut i = 0;
    loop {
        tokio::spawn(async move {
            // This `info!` call is extremely fast.
            // It just sends the message to the background thread
            // without blocking the async task.
            info!("I/O operation {} completed successfully.", i);
        });

        i += 1;
        // Simulate a high-throughput async environment
        if i > 1_000 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    info!("Main loop finished. Logs will continue to flush in the background.");
    // The program will wait here briefly to ensure final logs are written.
    tokio::time::sleep(Duration::from_secs(1)).await;
}
```

This setup gives you high-performance logging perfectly suited for an async environment. Your tight loop remains unblocked, and you don't have to manage file handles or threads manually.
