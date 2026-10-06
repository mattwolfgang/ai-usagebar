//! A self-hosted Ollama server — today's token tally, not a quota.
//!
//! Local models have no plan and no limit, so there is nothing to ask Ollama
//! for: the daemon keeps no usage history (`/api/ps` is only what is loaded
//! right now). The figures come from the two clients that do record it, each
//! restricted to the models the configured server lists in `/api/tags`:
//!
//! - **OpenCode** (`opencode::tally`) — every assistant message in its local
//!   `opencode.db` carries `tokens: {input, output, reasoning, cache: {read,
//!   write}}`. OpenCode's `input` excludes the prompt-cache hit, which on a
//!   long session is most of the prompt, so input here is `input + cache.read
//!   + cache.write` and output is `output + reasoning` — the sum is OpenCode's
//!   own `tokens.total`. Read-only; the database is never written.
//! - **Open WebUI** (`openwebui::tally`) — `GET /api/v1/analytics/tokens`,
//!   admin-only, which sums each assistant message's recorded `usage` per
//!   model between two epoch-second bounds (inclusive). Verified against the
//!   v0.11.4 router source.
//!
//! - **A usage proxy** (`proxy::tally`, `usage_proxy = true`) — when the
//!   server sits behind a proxy that records every generation and serves
//!   `GET /_usage?start=&end=`, that one figure replaces both sources above:
//!   it already includes their requests, plus every other client's.
//!
//! "Today" is the local calendar day. Any configured source failing fails the
//! whole refresh, so the bar shows the last complete total marked stale
//! rather than a partial sum that looks complete.

pub mod fetch;
pub mod opencode;
pub mod openwebui;
pub mod proxy;
pub mod vendor;

pub use fetch::{FetchOutcome, fetch_snapshot};

/// The form a model name is compared in: Ollama lists `name:latest` where
/// clients usually drop the default tag.
pub(crate) fn normalize_model(name: &str) -> &str {
    name.strip_suffix(":latest").unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::normalize_model;

    #[test]
    fn the_default_tag_is_dropped_and_others_kept() {
        assert_eq!(normalize_model("qwen3.5:9b"), "qwen3.5:9b");
        assert_eq!(
            normalize_model("qwen3.6-35b-a3b-32k:latest"),
            "qwen3.6-35b-a3b-32k"
        );
        assert_eq!(
            normalize_model("qwen3.6-35b-a3b-32k"),
            "qwen3.6-35b-a3b-32k"
        );
    }
}
