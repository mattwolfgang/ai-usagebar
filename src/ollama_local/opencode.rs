//! OpenCode's local session database, read-only.
//!
//! OpenCode (sst/opencode) keeps sessions in SQLite at
//! `$XDG_DATA_HOME/opencode/opencode.db` — XDG on every platform, because it
//! resolves the path with the `xdg-basedir` package. Each row of `message` is
//! one turn; `time_created` is epoch milliseconds and `data` is the turn as
//! JSON. Assistant turns look like:
//!
//! ```json
//! {"role":"assistant","providerID":"ollama","modelID":"qwen3.6-35b-a3b-32k",
//!  "tokens":{"total":20678,"input":177,"output":157,"reasoning":0,
//!            "cache":{"write":0,"read":20344}}, "...": "..."}
//! ```

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, params};

use crate::error::{AppError, Result};

/// One model's tally from OpenCode, before the server's model filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub model: String,
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// `$XDG_DATA_HOME/opencode/opencode.db`, else `~/.local/share/opencode/...`.
pub fn default_db_path() -> Result<PathBuf> {
    if let Some(data) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        let data = PathBuf::from(data);
        if data.is_absolute() {
            return Ok(data.join("opencode").join("opencode.db"));
        }
    }
    Ok(crate::cache::home_dir()?
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db"))
}

/// Assistant turns created in `[start_ms, end_ms)`, grouped by model.
pub fn tally(path: &Path, start_ms: i64, end_ms: i64) -> Result<Vec<Row>> {
    if !path.exists() {
        return Err(AppError::Other(format!(
            "OpenCode database not found at {}. Run OpenCode once, or set \
             `opencode = false` under [ollama-local].",
            path.display()
        )));
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| db_error(path, &e))?;
    let mut stmt = conn
        .prepare(
            "SELECT json_extract(data, '$.modelID') AS model,
                    count(*),
                    sum(coalesce(json_extract(data, '$.tokens.input'), 0)
                      + coalesce(json_extract(data, '$.tokens.cache.read'), 0)
                      + coalesce(json_extract(data, '$.tokens.cache.write'), 0)),
                    sum(coalesce(json_extract(data, '$.tokens.output'), 0)
                      + coalesce(json_extract(data, '$.tokens.reasoning'), 0))
               FROM message
              WHERE time_created >= ?1 AND time_created < ?2
                AND json_extract(data, '$.role') = 'assistant'
                AND json_extract(data, '$.tokens') IS NOT NULL
                AND model IS NOT NULL
              GROUP BY model",
        )
        .map_err(|e| db_error(path, &e))?;
    let rows = stmt
        .query_map(params![start_ms, end_ms], |row| {
            Ok(Row {
                model: row.get(0)?,
                requests: non_negative(row.get(1)?),
                input_tokens: non_negative(row.get(2)?),
                output_tokens: non_negative(row.get(3)?),
            })
        })
        .map_err(|e| db_error(path, &e))?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| db_error(path, &e))
}

fn non_negative(v: i64) -> u64 {
    u64::try_from(v).unwrap_or(0)
}

fn db_error(path: &Path, e: &rusqlite::Error) -> AppError {
    AppError::Other(format!(
        "could not read the OpenCode database at {}: {e}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn seed(dir: &TempDir, rows: &[(i64, &str)]) -> PathBuf {
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

    const QWEN: &str = r#"{"role":"assistant","modelID":"qwen","tokens":{"input":100,"output":20,"reasoning":5,"cache":{"read":1000,"write":3}}}"#;

    #[test]
    fn counts_cache_and_reasoning_within_the_window() {
        let dir = TempDir::new().unwrap();
        let path = seed(
            &dir,
            &[
                (1_000, QWEN),
                (1_500, QWEN),
                // Outside the window on both sides.
                (999, QWEN),
                (2_000, QWEN),
                // Not assistant turns.
                (1_200, r#"{"role":"user","modelID":"qwen"}"#),
                // An assistant turn still streaming carries no tokens yet.
                (1_300, r#"{"role":"assistant","modelID":"qwen"}"#),
            ],
        );
        let rows = tally(&path, 1_000, 2_000).unwrap();
        assert_eq!(
            rows,
            vec![Row {
                model: "qwen".into(),
                requests: 2,
                input_tokens: 2 * 1_103,
                output_tokens: 2 * 25,
            }]
        );
    }

    #[test]
    fn groups_by_model() {
        let dir = TempDir::new().unwrap();
        let other = r#"{"role":"assistant","modelID":"llama","tokens":{"input":1,"output":2}}"#;
        let path = seed(&dir, &[(10, QWEN), (11, other)]);
        let mut rows = tally(&path, 0, 100).unwrap();
        rows.sort_by(|a, b| a.model.cmp(&b.model));
        assert_eq!(rows[0].model, "llama");
        assert_eq!((rows[0].input_tokens, rows[0].output_tokens), (1, 2));
        assert_eq!(rows[1].model, "qwen");
    }

    #[test]
    fn a_missing_database_names_the_switch() {
        let dir = TempDir::new().unwrap();
        let err = tally(&dir.path().join("absent.db"), 0, 1).unwrap_err();
        assert!(err.to_string().contains("opencode = false"), "{err}");
    }
}
