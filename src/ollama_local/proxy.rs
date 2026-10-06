//! A usage-recording proxy in front of the Ollama server.
//!
//! Ollama keeps no usage history, so a proxy on the server's port forwards
//! every request unchanged and records each finished generation's counts.
//! Contract (a home Ollama + Open WebUI server, verified live 2026-10-06):
//!
//! ```json
//! GET /_usage?start=<epoch s>&end=<epoch s>     start inclusive, end exclusive
//! {"models":[{"model":"qwen3.6-35b-a3b-32k","requests":3,"input_tokens":18866,
//!             "output_tokens":182,"cached_input_tokens":13}]}
//! ```
//!
//! `input_tokens` is Ollama's own prompt count (`prompt_eval_count` /
//! `usage.prompt_tokens`), which already includes the cached prefix;
//! `cached_input_tokens` is that subset and is not added again. Rows are
//! bucketed by when the generation finished, and model names are as the
//! request named them.

use serde::Deserialize;

use crate::error::{AppError, Result};

#[derive(Debug, Deserialize)]
struct Body {
    models: Vec<Row>,
}

#[derive(Debug, Deserialize)]
pub struct Row {
    pub model: String,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// Usage per model finished in `[start, end)`, epoch seconds.
pub async fn tally(
    client: &reqwest::Client,
    server: &str,
    start: i64,
    end: i64,
) -> Result<Vec<Row>> {
    let url = format!(
        "{}/_usage?start={start}&end={end}",
        server.trim_end_matches('/')
    );
    let bytes = super::openwebui::get(client, &url, None, "Ollama usage proxy").await?;
    let body: Body = serde_json::from_slice(&bytes)
        .map_err(|e| AppError::Schema(format!("Ollama usage proxy response: {e}")))?;
    Ok(body.models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_missing_proxy_is_an_http_error() {
        let mut server = mockito::Server::new_async().await;
        // A plain Ollama answers an unknown path with 404.
        server
            .mock("GET", mockito::Matcher::Any)
            .with_status(404)
            .with_body("404 page not found")
            .create_async()
            .await;
        let err = tally(&reqwest::Client::new(), &server.url(), 0, 1)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::Http { status: 404, .. }), "{err}");
    }
}
