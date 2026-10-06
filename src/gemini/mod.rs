//! Gemini CLI — per-model request quota read with the Google session the
//! Gemini CLI saved after "Login with Google" (`creds.rs`), from the Cloud Code
//! `loadCodeAssist` + `retrieveUserQuota` calls the CLI itself makes for its
//! `/stats` quota view (`fetch.rs`, `types.rs`). API-key auth has no quota
//! surface, so for it the provider counts today's requests from the CLI's own
//! session logs instead (`local.rs`).

pub mod creds;
pub mod fetch;
pub mod local;
pub mod types;
pub mod vendor;

pub use fetch::{FetchOutcome, fetch_snapshot};
