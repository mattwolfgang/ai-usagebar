//! Today's tally: the server's model list, then each configured source,
//! cached as the finished snapshot. `Outcome` plumbing mirrors the other
//! single-shot providers: a fresh cache is reused, and a failed refresh falls
//! back to the last good payload through `outcome::fallback`.

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Local, NaiveDate, TimeZone, Utc};

use crate::cache::{Cache, acquire_lock_async};
use crate::config::OllamaLocalConfig;
use crate::error::{AppError, Result};
use crate::usage::{OllamaLocalModel, OllamaLocalSnapshot, OllamaLocalSource};

use super::normalize_model;

const LOCK_TIMEOUT: Duration = Duration::from_secs(15);

/// This vendor's [`Outcome`](crate::outcome::Outcome), specialised to its snapshot.
pub type FetchOutcome = crate::outcome::Outcome<OllamaLocalSnapshot>;

/// Everything a refresh reads, resolved from `[ollama-local]`.
#[derive(Debug, Clone)]
pub struct Sources {
    pub server: String,
    /// `None` when `opencode = false`.
    pub opencode_db: Option<PathBuf>,
    pub open_webui: Option<OpenWebUi>,
    /// Read the server's `/_usage` instead of the two client sources.
    pub usage_proxy: bool,
}

#[derive(Debug, Clone)]
pub struct OpenWebUi {
    pub url: String,
    pub api_key: String,
}

impl Sources {
    pub fn from_config(cfg: &OllamaLocalConfig) -> Result<Self> {
        if cfg.server.trim().is_empty() {
            return Err(AppError::Credentials(format!(
                "Ollama (local): no server. Set `server` under [ollama-local] in {}.",
                crate::config::config_path_hint()
            )));
        }
        if cfg.usage_proxy {
            return Ok(Self {
                server: cfg.server.clone(),
                opencode_db: None,
                open_webui: None,
                usage_proxy: true,
            });
        }
        let opencode_db = if cfg.opencode {
            Some(match &cfg.opencode_db {
                Some(path) => path.clone(),
                None => super::opencode::default_db_path()?,
            })
        } else {
            None
        };
        let open_webui = match &cfg.open_webui_url {
            None => None,
            Some(url) => {
                let api_key =
                    crate::config::optional_api_key(&cfg.api_key_env, cfg.api_key.as_deref())
                        .ok_or_else(|| {
                            AppError::Credentials(format!(
                                "Ollama (local): no Open WebUI API key. Export {} or set \
                             `api_key` under [ollama-local] in {} (an admin key, from \
                             Open WebUI's Settings > Account), or remove `open_webui_url`.",
                                cfg.api_key_env,
                                crate::config::config_path_hint()
                            ))
                        })?;
                Some(OpenWebUi {
                    url: url.clone(),
                    api_key,
                })
            }
        };
        Ok(Self {
            server: cfg.server.clone(),
            opencode_db,
            open_webui,
            usage_proxy: false,
        })
    }

    /// The host shown in the title row.
    fn server_label(&self) -> String {
        reqwest::Url::parse(&self.server)
            .ok()
            .and_then(|u| u.host_str().map(str::to_string))
            .unwrap_or_else(|| self.server.clone())
    }
}

/// The calendar day being counted, and its bounds as instants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Day {
    pub date: NaiveDate,
    pub start: DateTime<Utc>,
    /// Exclusive.
    pub end: DateTime<Utc>,
}

impl Day {
    /// The local calendar day containing `now`.
    pub fn local(now: DateTime<Utc>) -> Self {
        Self::in_zone(now, &Local)
    }

    pub fn in_zone<Tz: TimeZone>(now: DateTime<Utc>, tz: &Tz) -> Self {
        let date = now.with_timezone(tz).date_naive();
        let midnight = |d: NaiveDate| {
            let naive = d.and_hms_opt(0, 0, 0).expect("midnight exists");
            // A DST jump at midnight: take the earliest valid instant, or
            // fall back to UTC midnight rather than fail the refresh.
            tz.from_local_datetime(&naive)
                .earliest()
                .map(|t| t.with_timezone(&Utc))
                .unwrap_or_else(|| naive.and_utc())
        };
        let next = date.succ_opt().unwrap_or(date);
        Self {
            date,
            start: midnight(date),
            end: midnight(next),
        }
    }
}

pub async fn fetch_snapshot(
    client: &reqwest::Client,
    sources: &Sources,
    cache: &Cache,
    cache_ttl: Duration,
) -> Result<FetchOutcome> {
    fetch_snapshot_at(client, sources, cache, cache_ttl, Day::local(Utc::now())).await
}

pub async fn fetch_snapshot_at(
    client: &reqwest::Client,
    sources: &Sources,
    cache: &Cache,
    cache_ttl: Duration,
    day: Day,
) -> Result<FetchOutcome> {
    cache.ensure_dir()?;
    let _lock = acquire_lock_async(&cache.lock_path(), LOCK_TIMEOUT).await?;

    // Reused only while it still describes today: just after midnight the
    // cached total is yesterday's.
    if let Some(bytes) = cache.fresh_payload(cache_ttl)?
        && let Ok(snapshot) = parse_cache(&bytes)
        && snapshot.day == day.date
    {
        return Ok(crate::outcome::Outcome::cached(snapshot, cache, false));
    }

    match fetch_live(client, sources, day).await {
        Ok(snapshot) => {
            let bytes = serde_json::to_vec(&snapshot)
                .map_err(|e| AppError::Other(format!("ollama-local cache encode: {e}")))?;
            cache.write_payload(&bytes)?;
            Ok(crate::outcome::Outcome::fresh(snapshot))
        }
        Err(e) if e.is_transient() => crate::outcome::fallback(cache, None, e, parse_cache),
        Err(AppError::Http { status, body }) => {
            cache.mark_stale();
            let last_error = Some(cache.write_last_error(status, &body));
            crate::outcome::fallback(
                cache,
                last_error,
                AppError::Http { status, body },
                parse_cache,
            )
        }
        Err(e) => {
            cache.mark_stale();
            let last_error = Some(cache.write_last_error(0, &e.to_string()));
            crate::outcome::fallback(cache, last_error, e, parse_cache)
        }
    }
}

fn parse_cache(bytes: &[u8]) -> Result<OllamaLocalSnapshot> {
    serde_json::from_slice(bytes).map_err(|e| AppError::Schema(format!("ollama-local cache: {e}")))
}

async fn fetch_live(
    client: &reqwest::Client,
    sources: &Sources,
    day: Day,
) -> Result<OllamaLocalSnapshot> {
    if sources.usage_proxy {
        let rows = super::proxy::tally(
            client,
            &sources.server,
            day.start.timestamp(),
            day.end.timestamp(),
        )
        .await?;
        let models: Vec<OllamaLocalModel> = rows
            .into_iter()
            .map(|r| OllamaLocalModel {
                source: OllamaLocalSource::Server,
                model: normalize_model(&r.model).to_string(),
                requests: r.requests,
                input_tokens: r.input_tokens,
                output_tokens: r.output_tokens,
            })
            .collect();
        return Ok(finish(sources, day, models));
    }
    let served = super::openwebui::server_models(client, &sources.server).await?;
    let on_server = |model: &str| served.contains(normalize_model(model));

    let mut models = Vec::new();
    if let Some(path) = &sources.opencode_db {
        let rows = super::opencode::tally(
            path,
            day.start.timestamp_millis(),
            day.end.timestamp_millis(),
        )?;
        models.extend(
            rows.into_iter()
                .filter(|r| on_server(&r.model))
                .map(|r| OllamaLocalModel {
                    source: OllamaLocalSource::OpenCode,
                    model: normalize_model(&r.model).to_string(),
                    requests: r.requests,
                    input_tokens: r.input_tokens,
                    output_tokens: r.output_tokens,
                }),
        );
    }
    if let Some(owui) = &sources.open_webui {
        // Open WebUI's bounds are inclusive and in whole seconds.
        let usage = super::openwebui::tally(
            client,
            &owui.url,
            &owui.api_key,
            day.start.timestamp(),
            day.end.timestamp() - 1,
        )
        .await?;
        models.extend(
            usage
                .models
                .into_iter()
                .filter(|m| on_server(&m.model_id))
                .map(|m| OllamaLocalModel {
                    source: OllamaLocalSource::OpenWebUi,
                    model: normalize_model(&m.model_id).to_string(),
                    requests: m.message_count,
                    input_tokens: m.input_tokens,
                    output_tokens: m.output_tokens,
                }),
        );
    }
    Ok(finish(sources, day, models))
}

/// Drop empty rows, order largest first, and wrap the day's tally.
fn finish(sources: &Sources, day: Day, mut models: Vec<OllamaLocalModel>) -> OllamaLocalSnapshot {
    models.retain(|m| m.requests > 0 || m.total_tokens() > 0);
    models.sort_by(|a, b| {
        b.total_tokens()
            .cmp(&a.total_tokens())
            .then_with(|| a.model.cmp(&b.model))
    });
    OllamaLocalSnapshot {
        server: sources.server_label(),
        day: day.date,
        models,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;
    use rusqlite::{Connection, params};
    use tempfile::TempDir;

    /// 2026-10-06 in UTC-5: midnight is 05:00Z.
    fn day() -> Day {
        let now = Utc.with_ymd_and_hms(2026, 10, 6, 17, 0, 0).unwrap();
        Day::in_zone(now, &FixedOffset::west_opt(5 * 3600).unwrap())
    }

    fn seed_opencode(dir: &TempDir, rows: &[(i64, &str)]) -> PathBuf {
        let path = dir.path().join("opencode.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "CREATE TABLE message (id text PRIMARY KEY, session_id text NOT NULL,
             time_created integer NOT NULL, time_updated integer NOT NULL,
             data text NOT NULL);",
        )
        .unwrap();
        for (i, (at, data)) in rows.iter().enumerate() {
            conn.execute(
                "INSERT INTO message VALUES (?1, 's', ?2, ?2, ?3)",
                params![format!("m{i}"), at, data],
            )
            .unwrap();
        }
        path
    }

    fn cache(dir: &TempDir) -> Cache {
        let cache = Cache::at(dir.path().join("ollama-local"));
        cache.ensure_dir().unwrap();
        cache
    }

    const TAGS: &str =
        r#"{"models":[{"name":"qwen3.6-35b-a3b-32k:latest"},{"name":"qwen3.5:9b"}]}"#;

    #[test]
    fn day_bounds_follow_the_zone() {
        let d = day();
        assert_eq!(d.date, NaiveDate::from_ymd_opt(2026, 10, 6).unwrap());
        assert_eq!(d.start.to_rfc3339(), "2026-10-06T05:00:00+00:00");
        assert_eq!(d.end.to_rfc3339(), "2026-10-07T05:00:00+00:00");
    }

    #[tokio::test]
    async fn sums_both_sources_for_models_the_server_has() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", "/api/tags")
            .with_body(TAGS)
            .create_async()
            .await;
        let owui = server
            .mock("GET", "/api/v1/analytics/tokens")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded(
                    "start_date".into(),
                    day().start.timestamp().to_string(),
                ),
                mockito::Matcher::UrlEncoded(
                    "end_date".into(),
                    (day().end.timestamp() - 1).to_string(),
                ),
            ]))
            .match_header("Authorization", "Bearer test-key")
            .with_body(
                r#"{"models":[
                    {"model_id":"qwen3.5:9b","input_tokens":800,"output_tokens":200,"total_tokens":1000,"message_count":4},
                    {"model_id":"gpt-4o","input_tokens":9999,"output_tokens":9999,"total_tokens":19998,"message_count":9}
                ],"total_input_tokens":0,"total_output_tokens":0,"total_tokens":0}"#,
            )
            .create_async()
            .await;

        let dir = TempDir::new().unwrap();
        let start = day().start.timestamp_millis();
        let db = seed_opencode(
            &dir,
            &[
                (
                    start + 1_000,
                    r#"{"role":"assistant","modelID":"qwen3.6-35b-a3b-32k","tokens":{"input":100,"output":50,"reasoning":0,"cache":{"read":1850,"write":0}}}"#,
                ),
                // Yesterday: not counted.
                (
                    start - 1,
                    r#"{"role":"assistant","modelID":"qwen3.6-35b-a3b-32k","tokens":{"input":5,"output":5}}"#,
                ),
                // A model this server does not serve.
                (
                    start + 2_000,
                    r#"{"role":"assistant","modelID":"claude-sonnet","tokens":{"input":7,"output":7}}"#,
                ),
            ],
        );
        let sources = Sources {
            server: server.url(),
            opencode_db: Some(db),
            open_webui: Some(OpenWebUi {
                url: server.url(),
                api_key: "test-key".into(),
            }),
            usage_proxy: false,
        };

        let outcome = fetch_snapshot_at(
            &reqwest::Client::new(),
            &sources,
            &cache(&dir),
            Duration::from_secs(60),
            day(),
        )
        .await
        .unwrap();
        owui.assert_async().await;
        let snap = outcome.snapshot;
        assert!(!outcome.stale);
        assert_eq!(snap.day, day().date);
        assert_eq!(snap.source_tokens(OllamaLocalSource::OpenCode), 2_000);
        assert_eq!(snap.source_tokens(OllamaLocalSource::OpenWebUi), 1_000);
        assert_eq!(snap.total_tokens(), 3_000);
        assert_eq!(snap.total_requests(), 5);
        assert_eq!(snap.models[0].model, "qwen3.6-35b-a3b-32k");
    }

    #[tokio::test]
    async fn a_failing_source_serves_the_last_total_as_stale() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", "/api/tags")
            .with_body(TAGS)
            .create_async()
            .await;
        server
            .mock("GET", "/api/v1/analytics/tokens")
            .match_query(mockito::Matcher::Any)
            .with_status(401)
            .with_body(r#"{"detail":"Not authenticated"}"#)
            .create_async()
            .await;
        let dir = TempDir::new().unwrap();
        let cache = cache(&dir);
        let previous = OllamaLocalSnapshot {
            server: "ollama-box".into(),
            day: day().date,
            models: vec![OllamaLocalModel {
                source: OllamaLocalSource::OpenCode,
                model: "qwen3.5:9b".into(),
                requests: 1,
                input_tokens: 10,
                output_tokens: 5,
            }],
        };
        cache
            .write_payload(&serde_json::to_vec(&previous).unwrap())
            .unwrap();
        let sources = Sources {
            server: server.url(),
            opencode_db: None,
            open_webui: Some(OpenWebUi {
                url: server.url(),
                api_key: "bad".into(),
            }),
            usage_proxy: false,
        };

        let outcome = fetch_snapshot_at(
            &reqwest::Client::new(),
            &sources,
            &cache,
            Duration::ZERO,
            day(),
        )
        .await
        .unwrap();
        assert!(outcome.stale);
        assert_eq!(outcome.snapshot, previous);
        assert_eq!(outcome.last_error.map(|e| e.0), Some(401));
    }

    #[tokio::test]
    async fn yesterdays_fresh_cache_is_not_reused() {
        let mut server = mockito::Server::new_async().await;
        let tags = server
            .mock("GET", "/api/tags")
            .with_body(TAGS)
            .create_async()
            .await;
        let dir = TempDir::new().unwrap();
        let cache = cache(&dir);
        let yesterday = OllamaLocalSnapshot {
            server: "ollama-box".into(),
            day: day().date.pred_opt().unwrap(),
            models: vec![],
        };
        cache
            .write_payload(&serde_json::to_vec(&yesterday).unwrap())
            .unwrap();
        let db = seed_opencode(&dir, &[]);
        let sources = Sources {
            server: server.url(),
            opencode_db: Some(db),
            open_webui: None,
            usage_proxy: false,
        };
        let outcome = fetch_snapshot_at(
            &reqwest::Client::new(),
            &sources,
            &cache,
            Duration::from_secs(3600),
            day(),
        )
        .await
        .unwrap();
        tags.assert_async().await;
        assert_eq!(outcome.snapshot.day, day().date);
        assert_eq!(outcome.snapshot.total_tokens(), 0);
    }

    #[tokio::test]
    async fn the_usage_proxy_alone_is_read() {
        let mut server = mockito::Server::new_async().await;
        let usage = server
            .mock("GET", "/_usage")
            .match_query(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("start".into(), day().start.timestamp().to_string()),
                mockito::Matcher::UrlEncoded("end".into(), day().end.timestamp().to_string()),
            ]))
            .with_body(
                r#"{"models":[{"model":"qwen3.6-35b-a3b-32k:latest","requests":3,
                    "input_tokens":18866,"output_tokens":182,"cached_input_tokens":13}]}"#,
            )
            .create_async()
            .await;
        // Neither /api/tags nor any client source is consulted.
        let tags = server
            .mock("GET", "/api/tags")
            .expect(0)
            .create_async()
            .await;
        let dir = TempDir::new().unwrap();
        let sources = Sources {
            server: server.url(),
            opencode_db: None,
            open_webui: None,
            usage_proxy: true,
        };
        let outcome = fetch_snapshot_at(
            &reqwest::Client::new(),
            &sources,
            &cache(&dir),
            Duration::from_secs(60),
            day(),
        )
        .await
        .unwrap();
        usage.assert_async().await;
        tags.assert_async().await;
        let snap = outcome.snapshot;
        assert_eq!(snap.total_tokens(), 19_048);
        assert_eq!(snap.total_requests(), 3);
        assert_eq!(snap.models[0].source, OllamaLocalSource::Server);
        assert_eq!(snap.models[0].model, "qwen3.6-35b-a3b-32k");
    }

    #[test]
    fn proxy_mode_needs_no_open_webui_key_or_opencode_db() {
        let cfg = OllamaLocalConfig {
            enabled: true,
            server: "http://ollama-box:11434".into(),
            usage_proxy: true,
            opencode_db: Some(PathBuf::from("/nonexistent/opencode.db")),
            ..OllamaLocalConfig::default()
        };
        let sources = Sources::from_config(&cfg).unwrap();
        assert!(sources.usage_proxy);
        assert!(sources.opencode_db.is_none() && sources.open_webui.is_none());
    }

    #[test]
    fn open_webui_without_a_key_names_the_variable() {
        let cfg = OllamaLocalConfig {
            enabled: true,
            server: "http://ollama-box:11434".into(),
            open_webui_url: Some("http://ollama-box:3000".into()),
            // A name no environment sets, so the lookup is hermetic.
            api_key_env: "AI_USAGEBAR_TEST_UNSET_OPEN_WEBUI_KEY".into(),
            opencode_db: Some(PathBuf::from("/nonexistent/opencode.db")),
            ..OllamaLocalConfig::default()
        };
        let err = Sources::from_config(&cfg).unwrap_err();
        assert!(
            err.to_string()
                .contains("AI_USAGEBAR_TEST_UNSET_OPEN_WEBUI_KEY"),
            "{err}"
        );
    }

    #[test]
    fn the_title_shows_the_server_host() {
        let sources = Sources {
            server: "http://192.168.1.50:11434".into(),
            opencode_db: None,
            open_webui: None,
            usage_proxy: false,
        };
        assert_eq!(sources.server_label(), "192.168.1.50");
    }
}
