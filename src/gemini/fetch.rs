//! Orchestrates a Gemini CLI snapshot: read the CLI's saved Google session
//! (`creds.rs`), refresh it when it is about to expire and an OAuth client is
//! configured (the CLI's own file is never written back to), ask
//! `loadCodeAssist` which Code Assist project and tier the user has, then ask
//! `retrieveUserQuota` for that project's per-model buckets — the two calls the
//! Gemini CLI itself makes. Cache/stale/error-fallback shape mirrors
//! `kiro::fetch`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::cache::{Cache, acquire_lock_async, atomic_write};
use crate::error::{AppError, Result};
use crate::usage::{GeminiModelQuota, GeminiSnapshot};
use crate::vendor::{MAX_BODY_BYTES, read_body_capped};

use super::creds::{self, GeminiCredentials};
use super::types::{self, LoadCodeAssistResponse, QuotaResponse};

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
const REFRESH_TIMEOUT: Duration = Duration::from_secs(15);
const LOCK_TIMEOUT: Duration = Duration::from_secs(15);
const CODE_ASSIST_BASE: &str = "https://cloudcode-pa.googleapis.com/v1internal";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const OAUTH_CACHE_FILE: &str = "oauth.json";

/// Refresh this far ahead of the recorded expiry, as Kiro and Antigravity do.
const REFRESH_BUFFER_SECS: i64 = 300;

const SESSION_REJECTED: &str =
    "the Gemini CLI's Google session was rejected; run `gemini` to sign in again";
const SESSION_EXPIRED: &str = "the Gemini CLI's Google session expired; run `gemini` to renew it, \
     or set [gemini] oauth_client_id and oauth_client_secret so ai-usagebar can refresh it";
const REFRESH_REVOKED: &str =
    "the Gemini CLI's Google session was revoked; run `gemini` to sign in again";
const REFRESH_FAILED: &str = "Gemini token refresh failed";

#[derive(Debug, Clone)]
pub struct Endpoints {
    /// `https://cloudcode-pa.googleapis.com/v1internal`; a method is appended
    /// as `:<method>`, the CLI's own `getMethodUrl`.
    pub code_assist: String,
    pub token: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self {
            code_assist: CODE_ASSIST_BASE.to_string(),
            token: TOKEN_URL.to_string(),
        }
    }
}

impl Endpoints {
    fn method(&self, name: &str) -> String {
        format!("{}:{name}", self.code_assist.trim_end_matches('/'))
    }
}

/// The OAuth client a refresh needs. Config-only for the same reason as
/// `antigravity::cloud::OauthClient`: the Gemini CLI's installed-app client
/// secret is public by Google's definition (RFC 8252 §8.5), but a literal with
/// Google's client-secret prefix in a tracked file trips every secret scanner.
#[derive(Debug, Clone)]
pub struct OauthClient {
    pub id: String,
    pub secret: String,
}

impl OauthClient {
    pub fn from_config(id: Option<&str>, secret: Option<&str>) -> Option<Self> {
        let pick = |value: Option<&str>| {
            value
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        };
        Some(Self {
            id: pick(id)?,
            secret: pick(secret)?,
        })
    }
}

/// Everything a fetch needs besides the HTTP client and cache, resolved by the
/// caller from `[gemini]` (and, for `project`, `GOOGLE_CLOUD_PROJECT`) so this
/// module never reads config or the environment itself.
#[derive(Debug, Clone)]
pub struct Source {
    pub mode: crate::config::GeminiMode,
    pub creds_path: PathBuf,
    pub project: Option<String>,
    pub oauth: Option<OauthClient>,
    /// The CLI's session-log directory, for the local tally.
    pub logs_dir: PathBuf,
    pub daily_limits: std::collections::BTreeMap<String, u64>,
}

impl Source {
    /// Whether to read Google's quota (`true`) or tally the local logs.
    /// `auto` picks Google exactly when the CLI's Google login file exists:
    /// API-key users have none, and have no quota to ask Google for.
    pub fn uses_google(&self) -> bool {
        match self.mode {
            crate::config::GeminiMode::Google => true,
            crate::config::GeminiMode::Local => false,
            crate::config::GeminiMode::Auto => self.creds_path.exists(),
        }
    }
}

impl Source {
    /// Resolve `[gemini]` against the real home directory and environment.
    /// Production only; tests build a `Source` by hand.
    pub fn from_config(config: &crate::config::GeminiConfig) -> Result<Self> {
        let creds_path = creds::path_for(config)?;
        let env_project = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT_ID"]
            .iter()
            .find_map(|name| std::env::var(name).ok());
        let project = config
            .project
            .clone()
            .or(env_project)
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty());
        Ok(Self {
            mode: config.mode,
            creds_path,
            project,
            oauth: OauthClient::from_config(
                config.oauth_client_id.as_deref(),
                config.oauth_client_secret.as_deref(),
            ),
            logs_dir: super::local::logs_dir_for(config)?,
            daily_limits: config.daily_limits.clone(),
        })
    }
}

/// ai-usagebar's own record of a refreshed access token, scoped to the account
/// it was minted for.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct PersistedOAuth {
    account: String,
    access_token: String,
    expires_at: DateTime<Utc>,
}

/// This vendor's [`Outcome`](crate::outcome::Outcome).
pub type FetchOutcome = crate::outcome::Outcome<GeminiSnapshot>;

pub async fn fetch_snapshot(
    client: &reqwest::Client,
    source: &Source,
    cache: &Cache,
    cache_ttl: Duration,
) -> Result<FetchOutcome> {
    fetch_snapshot_at(
        client,
        source,
        cache,
        cache_ttl,
        &Endpoints::default(),
        Utc::now(),
    )
    .await
}

/// Endpoint and clock seam for tests.
pub async fn fetch_snapshot_at(
    client: &reqwest::Client,
    source: &Source,
    cache: &Cache,
    cache_ttl: Duration,
    endpoints: &Endpoints,
    now: DateTime<Utc>,
) -> Result<FetchOutcome> {
    // The local tally reads only this machine's files: no network, so no
    // cache, lock or fallback — it is always as fresh as the logs.
    if !source.uses_google() {
        let snap = super::local::snapshot_at(&source.logs_dir, &source.daily_limits, now)?;
        return Ok(crate::outcome::Outcome::fresh(snap));
    }

    cache.ensure_dir()?;
    let _lock = acquire_lock_async(&cache.lock_path(), LOCK_TIMEOUT).await?;

    // Resolve identity before accepting a cache hit, so a `gemini` re-login as
    // another account never keeps serving the previous one's figures.
    let creds = creds::read_from(&source.creds_path)?;

    if let Some(bytes) = cache.fresh_payload(cache_ttl)?
        && let Ok(snap) = parse_cache_at(&bytes, &creds.account_key, now)
    {
        return Ok(crate::outcome::Outcome::cached(snap, cache, false));
    }

    match fetch_live(client, endpoints, cache, source, &creds, now).await {
        Ok(snap) => {
            let bytes = serde_json::to_vec(&snap_to_json(&snap, &creds.account_key))?;
            cache.write_payload(&bytes)?;
            Ok(crate::outcome::Outcome::fresh(snap))
        }
        Err(e) if e.is_transient() => crate::outcome::fallback(cache, None, e, |bytes| {
            parse_cache_at(bytes, &creds.account_key, now)
        }),
        Err(e) => {
            cache.mark_stale();
            let last_error =
                error_to_pair(&e).map(|(code, msg)| cache.write_last_error(code, &msg));
            crate::outcome::fallback(cache, last_error, e, |bytes| {
                parse_cache_at(bytes, &creds.account_key, now)
            })
        }
    }
}

/// Auth failures carry a fixed message (set where they are raised); every
/// other error keeps its own text. Mirrors `kiro::fetch::error_to_pair`.
fn error_to_pair(e: &AppError) -> Option<(u16, String)> {
    match e {
        AppError::Http { status, body } => Some((*status, body.clone())),
        AppError::Credentials(msg) => Some((0, msg.clone())),
        e => Some((0, e.to_string())),
    }
}

fn needs_refresh(expires_at: Option<DateTime<Utc>>, now: DateTime<Utc>) -> bool {
    match expires_at {
        None => true,
        Some(at) => at.timestamp() < now.timestamp() + REFRESH_BUFFER_SECS,
    }
}

fn oauth_cache_path(cache: &Cache) -> PathBuf {
    cache.dir().join(OAUTH_CACHE_FILE)
}

/// The persisted token for exactly this account. Absent, unreadable, malformed
/// or another account's all read as `None`: the worst case is one extra
/// refresh round-trip.
fn read_persisted(path: &Path, account: &str) -> Option<PersistedOAuth> {
    let bytes = std::fs::read(path).ok()?;
    let persisted: PersistedOAuth = serde_json::from_slice(&bytes).ok()?;
    (persisted.account == account && !persisted.access_token.trim().is_empty()).then_some(persisted)
}

fn write_persisted(path: &Path, value: &PersistedOAuth) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    atomic_write(path, &bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| AppError::io_at(path, e))?;
    }
    Ok(())
}

/// Pick the access token to use: the later-expiring of the CLI's own and the
/// one this program last refreshed, refreshed again when it is about to die and
/// a client is configured. Without a client, a token that has not actually
/// expired yet is still used; one that has is a clear "run `gemini`".
async fn access_token(
    client: &reqwest::Client,
    endpoints: &Endpoints,
    cache: &Cache,
    source: &Source,
    creds: &GeminiCredentials,
    now: DateTime<Utc>,
) -> Result<String> {
    let path = oauth_cache_path(cache);
    let (mut token, mut expires_at) = (creds.access_token.clone(), creds.expires_at);
    if let Some(saved) = read_persisted(&path, &creds.account_key)
        && expires_at.is_none_or(|cli| saved.expires_at > cli)
    {
        token = saved.access_token;
        expires_at = Some(saved.expires_at);
    }
    if !needs_refresh(expires_at, now) {
        return Ok(token);
    }
    match (&source.oauth, creds.refresh_token.as_deref()) {
        (Some(oauth), Some(refresh_token)) => {
            let (access_token, expires_at) = tokio::time::timeout(
                REFRESH_TIMEOUT,
                refresh(client, &endpoints.token, oauth, refresh_token, now),
            )
            .await
            .map_err(|_| AppError::Transport("Gemini token refresh timed out".into()))??;
            let persisted = PersistedOAuth {
                account: creds.account_key.clone(),
                access_token,
                expires_at,
            };
            write_persisted(&path, &persisted)?;
            Ok(persisted.access_token)
        }
        // No way to refresh: a token with no recorded expiry may still be
        // good, and one inside the refresh buffer has a few minutes left.
        _ if expires_at.is_none_or(|at| at > now) => Ok(token),
        _ => Err(AppError::Credentials(SESSION_EXPIRED.into())),
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<serde_json::Value>,
}

/// Exchange the CLI's refresh token at Google's token endpoint. A definitive
/// 4xx means the session is gone; the upstream body is never surfaced.
async fn refresh(
    client: &reqwest::Client,
    token_url: &str,
    oauth: &OauthClient,
    refresh_token: &str,
    now: DateTime<Utc>,
) -> Result<(String, DateTime<Utc>)> {
    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", oauth.id.as_str()),
        ("client_secret", oauth.secret.as_str()),
    ];
    let resp = client
        .post(token_url)
        .header("Accept", "application/json")
        .form(&form)
        .send()
        .await?;
    let status = resp.status().as_u16();
    let body = read_body_capped(resp, MAX_BODY_BYTES).await?;
    if !(200..300).contains(&status) {
        if (400..500).contains(&status) && !matches!(status, 408 | 429) {
            return Err(AppError::Credentials(REFRESH_REVOKED.into()));
        }
        return Err(AppError::Http {
            status,
            body: REFRESH_FAILED.into(),
        });
    }
    let parsed: TokenResponse = serde_json::from_slice(&body)
        .map_err(|_| AppError::Schema("gemini token refresh response is malformed".into()))?;
    if parsed.access_token.trim().is_empty() {
        return Err(AppError::Schema(
            "gemini token refresh response has an empty access_token".into(),
        ));
    }
    let expires_in = match parsed.expires_in {
        Some(serde_json::Value::Number(n)) => n.as_i64(),
        Some(serde_json::Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    }
    .filter(|secs| (1..=366 * 24 * 60 * 60).contains(secs))
    .ok_or_else(|| AppError::Schema("gemini token refresh response: invalid expires_in".into()))?;
    let expires_at = now + chrono::Duration::seconds(expires_in);
    Ok((parsed.access_token, expires_at))
}

/// POST a Cloud Code method. 401/403 map to a fixed message: a rejected bearer
/// response can carry account detail.
async fn post(
    client: &reqwest::Client,
    url: &str,
    access_token: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value> {
    let resp = tokio::time::timeout(
        HTTP_TIMEOUT,
        client
            .post(url)
            .header("Authorization", format!("Bearer {access_token}"))
            .header("Accept", "application/json")
            .json(body)
            .send(),
    )
    .await
    .map_err(|_| AppError::Transport(format!("gemini timeout: {url}")))??;
    let status = resp.status().as_u16();
    if !(200..300).contains(&status) {
        let body = if matches!(status, 401 | 403) {
            SESSION_REJECTED.to_string()
        } else {
            format!("Gemini Code Assist API returned HTTP {status}")
        };
        return Err(AppError::Http { status, body });
    }
    let bytes = read_body_capped(resp, MAX_BODY_BYTES).await?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| AppError::Schema(format!("gemini response: {e}")))?;
    Ok(types::unwrap_response(value))
}

async fn fetch_live(
    client: &reqwest::Client,
    endpoints: &Endpoints,
    cache: &Cache,
    source: &Source,
    creds: &GeminiCredentials,
    now: DateTime<Utc>,
) -> Result<GeminiSnapshot> {
    let token = access_token(client, endpoints, cache, source, creds, now).await?;

    // The CLI's own `setupUser` request: client metadata plus the configured
    // project, if any. Never followed by `onboardUser` — see `types::project`.
    let load_body = serde_json::json!({
        "cloudaicompanionProject": source.project,
        "metadata": {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
            "duetProject": source.project,
        },
    });
    let load: LoadCodeAssistResponse = serde_json::from_value(
        post(
            client,
            &endpoints.method("loadCodeAssist"),
            &token,
            &load_body,
        )
        .await?,
    )
    .map_err(|e| AppError::Schema(format!("gemini loadCodeAssist response: {e}")))?;
    let project = load.project(source.project.as_deref())?;

    let quota: QuotaResponse = serde_json::from_value(
        post(
            client,
            &endpoints.method("retrieveUserQuota"),
            &token,
            &serde_json::json!({ "project": project }),
        )
        .await?,
    )
    .map_err(|e| AppError::Schema(format!("gemini retrieveUserQuota response: {e}")))?;
    types::to_snapshot(&quota, load.plan())
}

fn snap_to_json(snap: &GeminiSnapshot, account: &str) -> serde_json::Value {
    serde_json::json!({
        "account": account,
        "plan": snap.plan,
        "models": snap.models.iter().map(|m| serde_json::json!({
            "model": m.model,
            "utilization_pct": m.utilization_pct.unwrap_or(0),
            "remaining": m.remaining,
            "limit": m.limit,
            "resets_at": m.resets_at.map(|dt| dt.to_rfc3339()),
        })).collect::<Vec<_>>(),
    })
}

#[derive(Deserialize)]
struct CachedModel {
    model: String,
    utilization_pct: i32,
    remaining: Option<u64>,
    limit: Option<u64>,
    resets_at: Option<DateTime<Utc>>,
}

#[derive(Deserialize)]
struct CachedSnapshot {
    account: String,
    plan: String,
    models: Vec<CachedModel>,
}

/// A cache from another account, with no models, or with any bucket already
/// past its reset is not served: the figures it holds are known to be wrong.
fn parse_cache_at(bytes: &[u8], account: &str, now: DateTime<Utc>) -> Result<GeminiSnapshot> {
    let cached: CachedSnapshot = serde_json::from_slice(bytes)
        .map_err(|e| AppError::Schema(format!("gemini cache: {e}")))?;
    if cached.account != account {
        return Err(AppError::Schema(
            "gemini cache belongs to a different account; refetching".into(),
        ));
    }
    if cached.models.is_empty() {
        return Err(AppError::Schema("gemini cache has no models".into()));
    }
    if cached
        .models
        .iter()
        .any(|m| m.resets_at.is_some_and(|at| at <= now))
    {
        return Err(AppError::Schema(
            "gemini cache is past a quota reset; refetching".into(),
        ));
    }
    Ok(GeminiSnapshot {
        plan: cached.plan,
        source: crate::usage::GeminiSource::Google,
        models: cached
            .models
            .into_iter()
            .map(|m| GeminiModelQuota {
                model: m.model,
                utilization_pct: Some(m.utilization_pct.clamp(0, 100)),
                requests: None,
                tokens: None,
                remaining: m.remaining,
                limit: m.limit,
                resets_at: m.resets_at,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use tempfile::TempDir;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 3, 12, 0, 0).unwrap()
    }

    struct Fixture {
        _td: TempDir,
        cache: Cache,
        source: Source,
    }

    fn fixture(creds: serde_json::Value, oauth: bool) -> Fixture {
        let td = TempDir::new().unwrap();
        let creds_path = td.path().join("oauth_creds.json");
        std::fs::write(&creds_path, creds.to_string()).unwrap();
        let cache = Cache::at(td.path().join("cache").join("gemini"));
        cache.ensure_dir().unwrap();
        let source = Source {
            mode: crate::config::GeminiMode::Google,
            creds_path,
            project: None,
            oauth: oauth.then(|| OauthClient {
                id: "test-client".into(),
                secret: "test-secret".into(),
            }),
            logs_dir: td.path().join("logs"),
            daily_limits: Default::default(),
        };
        Fixture {
            _td: td,
            cache,
            source,
        }
    }

    fn live_creds() -> serde_json::Value {
        serde_json::json!({
            "access_token": "AT",
            "refresh_token": "RT",
            "expiry_date": (now() + chrono::Duration::hours(1)).timestamp_millis(),
        })
    }

    fn expired_creds() -> serde_json::Value {
        serde_json::json!({
            "access_token": "OLD",
            "refresh_token": "RT",
            "expiry_date": (now() - chrono::Duration::minutes(5)).timestamp_millis(),
        })
    }

    fn endpoints(server: &mockito::Server) -> Endpoints {
        Endpoints {
            code_assist: format!("{}/v1internal", server.url()),
            token: format!("{}/token", server.url()),
        }
    }

    const LOAD_BODY: &str = r#"{"currentTier":{"id":"free-tier","name":"Gemini Code Assist for individuals"},"cloudaicompanionProject":"proj-1"}"#;
    const QUOTA_BODY: &str = r#"{"buckets":[
        {"modelId":"gemini-2.5-pro","remainingFraction":0.4,"remainingAmount":"40","resetTime":"2026-10-04T07:00:00Z","tokenType":"REQUESTS"},
        {"modelId":"gemini-2.5-flash","remainingFraction":0.95,"resetTime":"2026-10-04T07:00:00Z","tokenType":"REQUESTS"}
    ]}"#;

    async fn mock_happy_path(
        server: &mut mockito::Server,
        token: &str,
    ) -> (mockito::Mock, mockito::Mock) {
        let load = server
            .mock("POST", "/v1internal:loadCodeAssist")
            .match_header("authorization", format!("Bearer {token}").as_str())
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "metadata": {"pluginType": "GEMINI"}
            })))
            .with_body(LOAD_BODY)
            .create_async()
            .await;
        let quota = server
            .mock("POST", "/v1internal:retrieveUserQuota")
            .match_header("authorization", format!("Bearer {token}").as_str())
            .match_body(mockito::Matcher::Json(
                serde_json::json!({"project": "proj-1"}),
            ))
            .with_body(QUOTA_BODY)
            .create_async()
            .await;
        (load, quota)
    }

    #[tokio::test]
    async fn fetches_project_then_quota_and_caches_it() {
        let mut server = mockito::Server::new_async().await;
        let (load, quota) = mock_happy_path(&mut server, "AT").await;
        let f = fixture(live_creds(), false);
        let client = reqwest::Client::new();

        let out = fetch_snapshot_at(
            &client,
            &f.source,
            &f.cache,
            Duration::from_secs(60),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        load.assert_async().await;
        quota.assert_async().await;
        assert!(!out.stale);
        assert_eq!(out.snapshot.plan, "Free");
        assert_eq!(out.snapshot.models[0].model, "gemini-2.5-pro");
        assert_eq!(out.snapshot.models[0].utilization_pct, Some(60));
        assert_eq!(out.snapshot.models[0].limit, Some(100));

        // Second call inside the TTL is served from cache without the network.
        let again = fetch_snapshot_at(
            &client,
            &f.source,
            &f.cache,
            Duration::from_secs(3600),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        assert_eq!(again.snapshot, out.snapshot);
        load.expect(1).assert_async().await;
    }

    #[tokio::test]
    async fn a_configured_project_is_sent_to_load_code_assist() {
        let mut server = mockito::Server::new_async().await;
        let load = server
            .mock("POST", "/v1internal:loadCodeAssist")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "cloudaicompanionProject": "my-proj",
                "metadata": {"duetProject": "my-proj"}
            })))
            .with_body(r#"{"currentTier":{"id":"standard-tier"}}"#)
            .create_async()
            .await;
        let quota = server
            .mock("POST", "/v1internal:retrieveUserQuota")
            .match_body(mockito::Matcher::Json(
                serde_json::json!({"project": "my-proj"}),
            ))
            .with_body(QUOTA_BODY)
            .create_async()
            .await;
        let mut f = fixture(live_creds(), false);
        f.source.project = Some("my-proj".into());
        fetch_snapshot_at(
            &reqwest::Client::new(),
            &f.source,
            &f.cache,
            Duration::from_secs(60),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        load.assert_async().await;
        quota.assert_async().await;
    }

    #[tokio::test]
    async fn an_expiring_token_is_refreshed_and_persisted_outside_the_cli_file() {
        let mut server = mockito::Server::new_async().await;
        let token = server
            .mock("POST", "/token")
            .match_body(mockito::Matcher::AllOf(vec![
                mockito::Matcher::UrlEncoded("grant_type".into(), "refresh_token".into()),
                mockito::Matcher::UrlEncoded("refresh_token".into(), "RT".into()),
                mockito::Matcher::UrlEncoded("client_id".into(), "test-client".into()),
            ]))
            .with_body(r#"{"access_token":"NEW","expires_in":3599}"#)
            .expect(1)
            .create_async()
            .await;
        let _ = mock_happy_path(&mut server, "NEW").await;
        let f = fixture(expired_creds(), true);
        let before = std::fs::read(&f.source.creds_path).unwrap();
        let client = reqwest::Client::new();

        fetch_snapshot_at(
            &client,
            &f.source,
            &f.cache,
            Duration::from_secs(0),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(&f.source.creds_path).unwrap(), before);
        let persisted = std::fs::read_to_string(oauth_cache_path(&f.cache)).unwrap();
        assert!(persisted.contains("NEW"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(oauth_cache_path(&f.cache))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        // The persisted token is reused on the next fetch: no second refresh.
        fetch_snapshot_at(
            &client,
            &f.source,
            &f.cache,
            Duration::from_secs(0),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        token.assert_async().await;
    }

    #[tokio::test]
    async fn an_expired_token_without_an_oauth_client_says_how_to_renew() {
        let server = mockito::Server::new_async().await;
        let f = fixture(expired_creds(), false);
        let err = fetch_snapshot_at(
            &reqwest::Client::new(),
            &f.source,
            &f.cache,
            Duration::from_secs(60),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::Credentials(ref m) if m.contains("run `gemini`")));
        assert!(err.to_string().contains("oauth_client_id"));
    }

    #[tokio::test]
    async fn a_token_inside_the_buffer_is_still_used_without_a_client() {
        let mut server = mockito::Server::new_async().await;
        let (load, _) = mock_happy_path(&mut server, "AT").await;
        let f = fixture(
            serde_json::json!({
                "access_token": "AT",
                "expiry_date": (now() + chrono::Duration::minutes(2)).timestamp_millis(),
            }),
            false,
        );
        fetch_snapshot_at(
            &reqwest::Client::new(),
            &f.source,
            &f.cache,
            Duration::from_secs(60),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        load.assert_async().await;
    }

    #[tokio::test]
    async fn a_revoked_refresh_token_is_a_credentials_error_without_the_body() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/token")
            .with_status(400)
            .with_body(r#"{"error":"invalid_grant","error_description":"account-detail"}"#)
            .create_async()
            .await;
        let f = fixture(expired_creds(), true);
        let err = fetch_snapshot_at(
            &reqwest::Client::new(),
            &f.source,
            &f.cache,
            Duration::from_secs(60),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::Credentials(_)));
        assert!(!err.to_string().contains("account-detail"));
    }

    #[tokio::test]
    async fn a_rejected_session_hides_the_body_and_falls_back_to_cache() {
        let mut server = mockito::Server::new_async().await;
        let (load, quota) = mock_happy_path(&mut server, "AT").await;
        let f = fixture(live_creds(), false);
        let client = reqwest::Client::new();
        let first = fetch_snapshot_at(
            &client,
            &f.source,
            &f.cache,
            Duration::from_secs(0),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        load.remove_async().await;
        quota.remove_async().await;

        server
            .mock("POST", "/v1internal:loadCodeAssist")
            .with_status(401)
            .with_body("secret account detail")
            .create_async()
            .await;
        let out = fetch_snapshot_at(
            &client,
            &f.source,
            &f.cache,
            Duration::from_secs(0),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap();
        assert!(out.stale);
        assert_eq!(out.snapshot, first.snapshot);
        // The cache owns what a 401 says once persisted (its shared
        // auth-failure wording); what matters here is that the body is gone.
        let (code, msg) = out.last_error.unwrap();
        assert_eq!(code, 401);
        assert!(!msg.is_empty());
        assert!(!msg.contains("secret"));
    }

    #[tokio::test]
    async fn a_failure_with_no_cache_returns_the_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1internal:loadCodeAssist")
            .with_status(500)
            .create_async()
            .await;
        let f = fixture(live_creds(), false);
        let err = fetch_snapshot_at(
            &reqwest::Client::new(),
            &f.source,
            &f.cache,
            Duration::from_secs(60),
            &endpoints(&server),
            now(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AppError::Http { status: 500, .. }));
    }

    #[test]
    fn cache_round_trips_and_rejects_other_accounts_and_past_resets() {
        let snap = GeminiSnapshot {
            plan: "Pro".into(),
            source: crate::usage::GeminiSource::Google,
            models: vec![GeminiModelQuota {
                model: "gemini-2.5-pro".into(),
                utilization_pct: Some(42),
                remaining: Some(58),
                limit: Some(100),
                requests: None,
                tokens: None,
                resets_at: Some(now() + chrono::Duration::hours(3)),
            }],
        };
        let bytes = serde_json::to_vec(&snap_to_json(&snap, "acct")).unwrap();
        assert_eq!(parse_cache_at(&bytes, "acct", now()).unwrap(), snap);
        assert!(parse_cache_at(&bytes, "other", now()).is_err());
        assert!(parse_cache_at(&bytes, "acct", now() + chrono::Duration::hours(4)).is_err());
    }

    #[test]
    fn auto_mode_uses_google_only_when_the_login_file_exists() {
        use crate::config::GeminiMode;
        let mut f = fixture(live_creds(), false);
        f.source.mode = GeminiMode::Auto;
        assert!(f.source.uses_google());
        std::fs::remove_file(&f.source.creds_path).unwrap();
        assert!(!f.source.uses_google());
        f.source.mode = GeminiMode::Google;
        assert!(f.source.uses_google());
        f.source.mode = GeminiMode::Local;
        std::fs::write(&f.source.creds_path, live_creds().to_string()).unwrap();
        assert!(!f.source.uses_google());
    }

    #[tokio::test]
    async fn an_api_key_user_gets_the_local_tally_without_the_network() {
        // No mocks at all: any HTTP call would fail the fetch.
        let server = mockito::Server::new_async().await;
        let mut f = fixture(live_creds(), false);
        std::fs::remove_file(&f.source.creds_path).unwrap();
        f.source.mode = crate::config::GeminiMode::Auto;
        f.source.daily_limits.insert("gemini-2.5-flash".into(), 250);
        let chats = f.source.logs_dir.join("proj").join("chats");
        std::fs::create_dir_all(&chats).unwrap();
        let line = serde_json::json!({
            "id": "m1", "type": "gemini", "model": "gemini-2.5-flash",
            "timestamp": Utc::now().to_rfc3339(), "tokens": {"total": 42},
        });
        std::fs::write(chats.join("session-x.jsonl"), format!("{line}\n")).unwrap();

        let out = fetch_snapshot_at(
            &reqwest::Client::new(),
            &f.source,
            &f.cache,
            Duration::from_secs(60),
            &endpoints(&server),
            Utc::now(),
        )
        .await
        .unwrap();
        assert_eq!(out.snapshot.source, crate::usage::GeminiSource::Local);
        let flash = &out.snapshot.models[0];
        assert_eq!(flash.requests, Some(1));
        assert_eq!(flash.tokens, Some(42));
        assert_eq!(flash.utilization_pct, Some(0));
        assert_eq!(flash.remaining, Some(249));
    }

    #[test]
    fn oauth_client_needs_both_halves() {
        assert!(OauthClient::from_config(Some("id"), None).is_none());
        assert!(OauthClient::from_config(Some(" "), Some("s")).is_none());
        let c = OauthClient::from_config(Some(" id "), Some(" s ")).unwrap();
        assert_eq!((c.id.as_str(), c.secret.as_str()), ("id", "s"));
    }

    #[test]
    fn method_urls_follow_the_cli_convention() {
        let e = Endpoints::default();
        assert_eq!(
            e.method("retrieveUserQuota"),
            "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota"
        );
    }
}
