// src/http/mod.rs
//
// HTTP module for enhanced client support

pub mod client;

pub use client::{
    EnhancedHttpClient, HttpClientConfig, HttpClientFactory,
};