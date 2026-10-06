//! Gemini CLI's own Google OAuth session, read from `~/.gemini/oauth_creds.json`.
//!
//! The CLI writes that file after "Login with Google" (google-auth-library's
//! `Credentials` shape: `access_token`, `refresh_token`, `expiry_date` in epoch
//! milliseconds, and an `id_token`). It is read, never written: a refreshed
//! token goes to ai-usagebar's own vendor cache (see `fetch.rs`), exactly as
//! Kiro and Antigravity do with their CLIs' sessions.
//!
//! Two other ways the CLI can authenticate leave nothing here to read, and each
//! gets its own message rather than a generic "not found":
//! `GEMINI_FORCE_ENCRYPTED_FILE_STORAGE=true` moves the session into the OS
//! keyring or an encrypted file, and API-key auth has no quota surface at all.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{AppError, Result};

const LOGIN_HINT: &str = "run `gemini`, choose \"Login with Google\", then try again \
    — or, for an API key, set [gemini] mode = \"local\" to count usage from the CLI's logs";

/// `~/.gemini/oauth_creds.json` — the CLI's `Storage.getOAuthCredsPath()`.
pub fn default_path() -> Result<PathBuf> {
    Ok(crate::cache::home_dir()?
        .join(".gemini")
        .join("oauth_creds.json"))
}

/// `[gemini] creds_path` when set, else [`default_path`].
pub fn path_for(config: &crate::config::GeminiConfig) -> Result<PathBuf> {
    match &config.creds_path {
        Some(path) => Ok(path.clone()),
        None => default_path(),
    }
}

/// The parts of the CLI's session this program uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiCredentials {
    pub access_token: String,
    pub refresh_token: Option<String>,
    /// `None` when the file records no expiry; treated as needing refresh.
    pub expires_at: Option<DateTime<Utc>>,
    /// Stable, non-reversible identity of the signed-in account. Keys the
    /// cache so a `gemini` re-login as someone else never serves the previous
    /// account's figures.
    pub account_key: String,
}

#[derive(Debug, Deserialize)]
struct RawCreds {
    #[serde(rename = "type")]
    kind: Option<String>,
    access_token: Option<String>,
    refresh_token: Option<String>,
    expiry_date: Option<serde_json::Value>,
    id_token: Option<String>,
}

/// Read and validate the CLI's credential file at `path`.
pub fn read_from(path: &Path) -> Result<GeminiCredentials> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(AppError::Credentials(format!(
                "no Gemini CLI Google login found at {}; {LOGIN_HINT}",
                crate::display::sanitize_untrusted_path(path)
            )));
        }
        Err(e) => return Err(AppError::io_at(path, e)),
    };
    parse(&bytes)
}

fn parse(bytes: &[u8]) -> Result<GeminiCredentials> {
    // Never echo serde's message: it can quote a fragment of the file.
    let raw: RawCreds = serde_json::from_slice(bytes).map_err(|_| {
        AppError::Credentials(format!(
            "the Gemini CLI's oauth_creds.json is not valid JSON; {LOGIN_HINT}"
        ))
    })?;
    // The CLI also accepts Application Default Credentials (a service account
    // or external account) through the same loader. Those authenticate a
    // project, not a Code Assist user, and carry no per-user quota.
    if let Some(kind) = raw.kind.as_deref()
        && kind != "authorized_user"
    {
        return Err(AppError::Credentials(
            "the Gemini CLI is signed in with Application Default Credentials, which have no \
             per-user Gemini quota; sign in with \"Login with Google\" instead"
                .into(),
        ));
    }
    let access_token = raw
        .access_token
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            AppError::Credentials(format!(
                "the Gemini CLI's oauth_creds.json has no access token; {LOGIN_HINT}"
            ))
        })?;
    let refresh_token = raw
        .refresh_token
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty());
    let expires_at = raw.expiry_date.as_ref().and_then(expiry_from_value);
    let account_key = account_key(
        raw.id_token.as_deref(),
        refresh_token.as_deref(),
        &access_token,
    );
    Ok(GeminiCredentials {
        access_token,
        refresh_token,
        expires_at,
        account_key,
    })
}

/// google-auth-library stores `expiry_date` as epoch milliseconds. Accept a
/// stringified number too, and reject anything outside chrono's range.
fn expiry_from_value(value: &serde_json::Value) -> Option<DateTime<Utc>> {
    let millis = match value {
        serde_json::Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64))?,
        serde_json::Value::String(s) => s.trim().parse::<i64>().ok()?,
        _ => return None,
    };
    (millis > 0)
        .then_some(millis)
        .and_then(DateTime::from_timestamp_millis)
}

/// The id token's `sub` is the Google account's stable id, so it survives a
/// refresh-token rotation; the refresh token, then the access token, stand in
/// when there is no id token. Only a truncated digest is kept.
fn account_key(id_token: Option<&str>, refresh_token: Option<&str>, access_token: &str) -> String {
    let sub = id_token
        .and_then(crate::jwt::claims)
        .and_then(|claims| claims.get("sub")?.as_str().map(str::to_string))
        .filter(|sub| !sub.is_empty());
    let basis = match (&sub, refresh_token) {
        (Some(sub), _) => format!("sub:{sub}"),
        (None, Some(rt)) => format!("rt:{rt}"),
        (None, None) => format!("at:{access_token}"),
    };
    fingerprint(&basis)
}

pub(crate) fn fingerprint(secret: &str) -> String {
    let digest = Sha256::digest(secret.as_bytes());
    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use tempfile::TempDir;

    fn id_token(sub: &str) -> String {
        let enc = |v: serde_json::Value| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(v.to_string())
        };
        format!(
            "{}.{}.sig",
            enc(serde_json::json!({"alg": "RS256"})),
            enc(serde_json::json!({"sub": sub}))
        )
    }

    #[test]
    fn reads_the_cli_file_shape() {
        let body = serde_json::json!({
            "access_token": "at",
            "refresh_token": "rt",
            "scope": "openid",
            "token_type": "Bearer",
            "id_token": id_token("12345"),
            "expiry_date": 1_790_000_000_000_i64,
        });
        let creds = parse(body.to_string().as_bytes()).unwrap();
        assert_eq!(creds.access_token, "at");
        assert_eq!(creds.refresh_token.as_deref(), Some("rt"));
        assert_eq!(
            creds.expires_at,
            DateTime::from_timestamp_millis(1_790_000_000_000)
        );
        assert_eq!(creds.account_key, fingerprint("sub:12345"));
    }

    #[test]
    fn account_key_follows_the_google_account_not_the_refresh_token() {
        let a = account_key(Some(&id_token("1")), Some("rt-a"), "at");
        let b = account_key(Some(&id_token("1")), Some("rt-b"), "at");
        let c = account_key(Some(&id_token("2")), Some("rt-a"), "at");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(!a.contains("rt"));
    }

    #[test]
    fn account_key_falls_back_without_an_id_token() {
        assert_eq!(account_key(None, Some("rt"), "at"), fingerprint("rt:rt"));
        assert_eq!(account_key(None, None, "at"), fingerprint("at:at"));
        assert_eq!(
            account_key(Some("not-a-jwt"), Some("rt"), "at"),
            fingerprint("rt:rt")
        );
    }

    #[test]
    fn stringified_and_missing_expiry_are_handled() {
        let s = parse(br#"{"access_token":"at","expiry_date":"1790000000000"}"#).unwrap();
        assert!(s.expires_at.is_some());
        let none = parse(br#"{"access_token":"at"}"#).unwrap();
        assert_eq!(none.expires_at, None);
        assert_eq!(none.refresh_token, None);
        let bad = parse(br#"{"access_token":"at","expiry_date":true}"#).unwrap();
        assert_eq!(bad.expires_at, None);
    }

    #[test]
    fn missing_access_token_is_a_login_hint() {
        let err = parse(br#"{"refresh_token":"rt"}"#).unwrap_err();
        assert!(matches!(err, AppError::Credentials(ref m) if m.contains("Login with Google")));
    }

    #[test]
    fn adc_service_accounts_are_refused_and_authorized_user_is_fine() {
        let err = parse(br#"{"type":"service_account","access_token":"at"}"#).unwrap_err();
        assert!(matches!(err, AppError::Credentials(ref m) if m.contains("Application Default")));
        assert!(parse(br#"{"type":"authorized_user","access_token":"at"}"#).is_ok());
    }

    #[test]
    fn malformed_json_does_not_echo_the_file() {
        let err = parse(b"{\"access_token\": \"secret-ish").unwrap_err();
        assert!(!err.to_string().contains("secret-ish"));
    }

    #[test]
    fn a_missing_file_tells_the_user_how_to_log_in() {
        let td = TempDir::new().unwrap();
        let err = read_from(&td.path().join("oauth_creds.json")).unwrap_err();
        assert!(matches!(err, AppError::Credentials(ref m) if m.contains("Login with Google")));
    }

    #[test]
    fn reads_from_an_injected_path() {
        let td = TempDir::new().unwrap();
        let path = td.path().join("oauth_creds.json");
        std::fs::write(&path, br#"{"access_token":"at","refresh_token":"rt"}"#).unwrap();
        let creds = read_from(&path).unwrap();
        assert_eq!(creds.access_token, "at");
    }
}
