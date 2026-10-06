//! Open WebUI's admin analytics and the Ollama server's model list.
//!
//! Open WebUI (v0.11.4, `backend/open_webui/routers/analytics.py`):
//!
//! ```json
//! GET /api/v1/analytics/tokens?start_date=<epoch s>&end_date=<epoch s>
//! Authorization: Bearer <admin API key>
//! {"models":[{"model_id":"qwen3.5:9b","input_tokens":812,"output_tokens":95,
//!             "total_tokens":907,"message_count":3}],
//!  "total_input_tokens":812,"total_output_tokens":95,"total_tokens":907}
//! ```
//!
//! Both bounds are inclusive (`created_at >= start AND created_at <= end`).
//! A non-admin key gets 401.

use std::collections::BTreeSet;
use std::time::Duration;

use serde::Deserialize;

use crate::error::{AppError, Result};

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Deserialize)]
pub struct TokenUsage {
    pub models: Vec<ModelTokens>,
}

#[derive(Debug, Deserialize)]
pub struct ModelTokens {
    pub model_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub message_count: u64,
}

#[derive(Debug, Deserialize)]
struct Tags {
    models: Vec<Tag>,
}

#[derive(Debug, Deserialize)]
struct Tag {
    name: String,
}

/// Token usage per model with `created_at` in `[start, end]`, epoch seconds.
pub async fn tally(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    start: i64,
    end: i64,
) -> Result<TokenUsage> {
    let url = format!(
        "{}/api/v1/analytics/tokens?start_date={start}&end_date={end}",
        base_url.trim_end_matches('/')
    );
    let bytes = get(client, &url, Some(api_key), "Open WebUI").await?;
    serde_json::from_slice(&bytes)
        .map_err(|e| AppError::Schema(format!("Open WebUI analytics response: {e}")))
}

/// The models the Ollama server has, normalized (see `normalize_model`).
pub async fn server_models(client: &reqwest::Client, server: &str) -> Result<BTreeSet<String>> {
    let url = format!("{}/api/tags", server.trim_end_matches('/'));
    let bytes = get(client, &url, None, "Ollama").await?;
    let tags: Tags = serde_json::from_slice(&bytes)
        .map_err(|e| AppError::Schema(format!("Ollama /api/tags response: {e}")))?;
    Ok(tags
        .models
        .iter()
        .map(|t| super::normalize_model(&t.name).to_string())
        .collect())
}

pub(super) async fn get(
    client: &reqwest::Client,
    url: &str,
    bearer: Option<&str>,
    what: &str,
) -> Result<Vec<u8>> {
    let mut req = client.get(url).header("Accept", "application/json");
    if let Some(key) = bearer {
        req = req.header("Authorization", format!("Bearer {key}"));
    }
    let resp = tokio::time::timeout(HTTP_TIMEOUT, req.send())
        .await
        .map_err(|_| AppError::Transport(format!("{what} timeout: {url}")))??;
    let status = resp.status();
    let bytes = crate::vendor::read_body_capped(resp, crate::vendor::MAX_BODY_BYTES).await?;
    if !status.is_success() {
        let body: String = String::from_utf8_lossy(&bytes).chars().take(200).collect();
        return Err(AppError::Http {
            status: status.as_u16(),
            body: format!("{what}: {body}"),
        });
    }
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn analytics_sends_the_key_and_both_bounds() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("GET", "/api/v1/analytics/tokens")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("start_date".into(), "100".into()),
                mockito::Matcher::UrlEncoded("end_date".into(), "199".into()),
            ]))
            .match_header("Authorization", "Bearer test-key")
            .with_body(
                r#"{"models":[{"model_id":"qwen3.5:9b","input_tokens":812,"output_tokens":95,
                    "total_tokens":907,"message_count":3}],
                    "total_input_tokens":812,"total_output_tokens":95,"total_tokens":907}"#,
            )
            .create_async()
            .await;
        let usage = tally(
            &reqwest::Client::new(),
            &format!("{}/", server.url()),
            "test-key",
            100,
            199,
        )
        .await
        .unwrap();
        mock.assert_async().await;
        assert_eq!(usage.models[0].model_id, "qwen3.5:9b");
        assert_eq!(usage.models[0].message_count, 3);
    }

    #[tokio::test]
    async fn a_non_admin_key_is_an_http_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", mockito::Matcher::Any)
            .with_status(401)
            .with_body(r#"{"detail":"Not authenticated"}"#)
            .create_async()
            .await;
        let err = tally(&reqwest::Client::new(), &server.url(), "k", 0, 1)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::Http { status: 401, .. }), "{err}");
    }

    #[tokio::test]
    async fn server_models_are_normalized() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", "/api/tags")
            .with_body(r#"{"models":[{"name":"qwen3.5:9b"},{"name":"big-32k:latest"}]}"#)
            .create_async()
            .await;
        let models = server_models(&reqwest::Client::new(), &server.url())
            .await
            .unwrap();
        assert!(models.contains("qwen3.5:9b"));
        assert!(models.contains("big-32k"));
    }
}
