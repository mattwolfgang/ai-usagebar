//! Today's Gemini usage, counted from the Gemini CLI's own session logs.
//!
//! An API key reaches the Gemini API's inference routes and nothing else:
//! Google publishes no quota or usage endpoint for it, and the per-key limits
//! live only in the AI Studio and Cloud consoles. What the machine does have is
//! the CLI's own record of every response it received, under
//! `~/.gemini/tmp/<project>/chats/session-*.jsonl` (older CLIs wrote a single
//! `session-*.json`). Each model response there carries its `model`, its
//! `tokens`, a `timestamp` and an `id`; that is all this module reads. Message
//! text is parsed past, never kept.
//!
//! One response is one request. The day is Google's quota day — midnight to
//! midnight Pacific time — so a configured daily limit lines up with the
//! limit Google enforces. The count is a floor: requests made with the same
//! key from another app or machine are not in these logs.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Utc};
use serde::Deserialize;

use crate::error::{AppError, Result};
use crate::usage::{GeminiModelQuota, GeminiSnapshot, GeminiSource};

/// Bounds, in the spirit of `context::scan_dir`: a pathological log directory
/// costs a partial count, never a hung or memory-hungry widget.
const MAX_WALK_ENTRIES: usize = 10_000;
const MAX_FILES: usize = 500;
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

pub const PLAN_LABEL: &str = "API key";

/// `~/.gemini/tmp` — the CLI's `Storage.getGlobalTempDir()`.
pub fn default_logs_dir() -> Result<PathBuf> {
    Ok(crate::cache::home_dir()?.join(".gemini").join("tmp"))
}

/// `[gemini] logs_dir` when set, else [`default_logs_dir`].
pub fn logs_dir_for(config: &crate::config::GeminiConfig) -> Result<PathBuf> {
    match &config.logs_dir {
        Some(path) => Ok(path.clone()),
        None => default_logs_dir(),
    }
}

/// Whether any session log exists under `root`, without reading one. For
/// `detect`, which must stay cheap.
pub fn has_session_logs(root: &Path) -> bool {
    let Ok(projects) = std::fs::read_dir(root) else {
        return false;
    };
    projects.flatten().take(MAX_WALK_ENTRIES).any(|project| {
        std::fs::read_dir(project.path().join("chats"))
            .map(|mut files| files.any(|f| f.is_ok_and(|f| is_session_file(&f.path()))))
            .unwrap_or(false)
    })
}

fn is_session_file(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.starts_with("session-") && (name.ends_with(".jsonl") || name.ends_with(".json"))
}

// ── Pacific time ─────────────────────────────────────────────────────────────
//
// Google's daily quotas reset at midnight Pacific. No timezone crate is a
// dependency, and US DST has had one rule since 2007: PDT (UTC-7) from 02:00
// local on the second Sunday of March to 02:00 local on the first Sunday of
// November, PST (UTC-8) otherwise.

fn nth_sunday(year: i32, month: u32, n: u32) -> NaiveDate {
    let first = NaiveDate::from_ymd_opt(year, month, 1).expect("valid month");
    let offset = (7 - first.weekday().num_days_from_sunday()) % 7;
    first + Duration::days(i64::from(offset + 7 * (n - 1)))
}

/// UTC offset of US Pacific time at `instant`, in hours (-7 or -8).
fn pacific_offset_hours(instant: DateTime<Utc>) -> i64 {
    let year = instant.year();
    // 02:00 PST on the DST start date is 10:00 UTC; 02:00 PDT on the end date
    // is 09:00 UTC.
    let start = nth_sunday(year, 3, 2)
        .and_hms_opt(10, 0, 0)
        .expect("valid time");
    let end = nth_sunday(year, 11, 1)
        .and_hms_opt(9, 0, 0)
        .expect("valid time");
    let naive = instant.naive_utc();
    if naive >= start && naive < end {
        -7
    } else {
        -8
    }
}

/// The Pacific calendar day containing `now`, as a UTC `[start, end)` range.
pub fn pacific_day(now: DateTime<Utc>) -> (DateTime<Utc>, DateTime<Utc>) {
    let local_date = (now + Duration::hours(pacific_offset_hours(now))).date_naive();
    let midnight_utc = |date: NaiveDate| {
        // Midnight is never inside a DST transition (those happen at 02:00),
        // so the offset at a rough guess of the instant is the right one.
        let guess = Utc.from_utc_datetime(&date.and_hms_opt(8, 0, 0).expect("valid time"));
        guess - Duration::hours(8 + pacific_offset_hours(guess))
    };
    let start = midnight_utc(local_date);
    let end = midnight_utc(local_date + Duration::days(1));
    (start, end)
}

// ── Log parsing ──────────────────────────────────────────────────────────────

/// The only fields read from a message. Everything else is skipped by serde.
#[derive(Debug, Deserialize)]
struct Message {
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    model: Option<String>,
    timestamp: Option<String>,
    tokens: Option<Tokens>,
}

#[derive(Debug, Deserialize)]
struct Tokens {
    total: Option<u64>,
}

/// A `$set` patch carrying the session's message list, or the legacy single
/// `.json` document with the same list at its root.
#[derive(Debug, Deserialize)]
struct WithMessages {
    #[serde(default)]
    messages: Vec<Message>,
}

#[derive(Debug, Deserialize)]
struct SetPatch {
    #[serde(rename = "$set")]
    set: WithMessages,
}

/// One counted response, keyed by message id so a message written several
/// times (it is rewritten as it streams) is counted once, with its last tokens.
struct Response {
    model: String,
    tokens: u64,
}

fn record(
    seen: &mut HashMap<String, Response>,
    msg: Message,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) {
    if msg.kind.as_deref() != Some("gemini") {
        return;
    }
    let (Some(id), Some(model), Some(timestamp), Some(tokens)) =
        (msg.id, msg.model, msg.timestamp, msg.tokens)
    else {
        return;
    };
    let model = model.trim();
    if model.is_empty() {
        return;
    }
    let Ok(at) = DateTime::parse_from_rfc3339(timestamp.trim()) else {
        return;
    };
    let at = at.with_timezone(&Utc);
    if at < start || at >= end {
        return;
    }
    seen.insert(
        id,
        Response {
            model: model.to_string(),
            tokens: tokens.total.unwrap_or(0),
        },
    );
}

fn read_jsonl(
    path: &Path,
    seen: &mut HashMap<String, Response>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) {
    let Ok(file) = std::fs::File::open(path) else {
        return;
    };
    let mut reader = BufReader::new(file.take(MAX_FILE_BYTES));
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if line.len() > MAX_LINE_BYTES {
            continue;
        }
        // A line is either a single message or a `$set` patch carrying the
        // whole list; anything else (session metadata) parses as neither.
        if let Ok(patch) = serde_json::from_slice::<SetPatch>(&line) {
            for msg in patch.set.messages {
                record(seen, msg, start, end);
            }
        } else if let Ok(msg) = serde_json::from_slice::<Message>(&line) {
            record(seen, msg, start, end);
        }
    }
}

fn read_json(
    path: &Path,
    seen: &mut HashMap<String, Response>,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) {
    let Ok(file) = std::fs::File::open(path) else {
        return;
    };
    let mut bytes = Vec::new();
    if file.take(MAX_FILE_BYTES).read_to_end(&mut bytes).is_err() {
        return;
    }
    if let Ok(doc) = serde_json::from_slice::<WithMessages>(&bytes) {
        for msg in doc.messages {
            record(seen, msg, start, end);
        }
    }
}

/// Per-model `(requests, tokens)` for responses timestamped in `[start, end)`,
/// from every session log under `root`. A log whose modification time is
/// before `start` cannot hold a response from today and is not opened.
pub fn tally_dir(
    root: &Path,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<BTreeMap<String, (u64, u64)>> {
    let projects = std::fs::read_dir(root).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AppError::Credentials(format!(
                "no Gemini CLI session logs at {}; run `gemini` once, then try again",
                crate::display::sanitize_untrusted_path(root)
            ))
        } else {
            AppError::io_at(root, e)
        }
    })?;

    let start_time: std::time::SystemTime = start.into();
    let mut files = Vec::new();
    let mut visited = 0usize;
    'walk: for project in projects.flatten() {
        let Ok(entries) = std::fs::read_dir(project.path().join("chats")) else {
            continue;
        };
        for entry in entries.flatten() {
            visited += 1;
            if visited > MAX_WALK_ENTRIES || files.len() >= MAX_FILES {
                break 'walk;
            }
            let path = entry.path();
            if !is_session_file(&path) {
                continue;
            }
            let fresh = entry
                .metadata()
                .ok()
                .filter(|m| m.is_file())
                .and_then(|m| m.modified().ok())
                .is_some_and(|modified| modified >= start_time);
            if fresh {
                files.push(path);
            }
        }
    }

    let mut seen = HashMap::new();
    for path in &files {
        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            read_jsonl(path, &mut seen, start, end);
        } else {
            read_json(path, &mut seen, start, end);
        }
    }

    let mut totals: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for response in seen.into_values() {
        let entry = totals.entry(response.model).or_default();
        entry.0 += 1;
        entry.1 = entry.1.saturating_add(response.tokens);
    }
    Ok(totals)
}

/// Build a snapshot from a tally and the user's configured daily request
/// limits. A limited model shows even on a day it was not used (at 0%); an
/// unlimited one shows only once it has been used.
pub fn to_snapshot(
    totals: BTreeMap<String, (u64, u64)>,
    limits: &BTreeMap<String, u64>,
    resets_at: DateTime<Utc>,
) -> GeminiSnapshot {
    let mut models: BTreeMap<String, (u64, u64)> = totals;
    for model in limits.keys() {
        models.entry(model.clone()).or_default();
    }
    let models = models
        .into_iter()
        .map(|(model, (requests, tokens))| {
            let limit = limits.get(&model).copied().filter(|limit| *limit > 0);
            let utilization_pct = limit.map(|limit| {
                let pct = (u128::from(requests) * 100 + u128::from(limit) / 2) / u128::from(limit);
                pct.min(9999) as i32
            });
            GeminiModelQuota {
                remaining: limit.map(|limit| limit.saturating_sub(requests)),
                limit,
                utilization_pct,
                requests: Some(requests),
                tokens: Some(tokens),
                resets_at: Some(resets_at),
                model,
            }
        })
        .collect();
    let mut snap = GeminiSnapshot {
        plan: PLAN_LABEL.to_string(),
        source: GeminiSource::Local,
        models,
    };
    snap.sort();
    snap
}

/// Tally today's usage under `root` at `now`.
pub fn snapshot_at(
    root: &Path,
    limits: &BTreeMap<String, u64>,
    now: DateTime<Utc>,
) -> Result<GeminiSnapshot> {
    let (start, end) = pacific_day(now);
    let totals = tally_dir(root, start, end)?;
    Ok(to_snapshot(totals, limits, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn utc(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    #[test]
    fn pacific_offset_follows_us_dst() {
        // 2026: DST from Mar 8 10:00 UTC to Nov 1 09:00 UTC.
        assert_eq!(pacific_offset_hours(utc("2026-03-08T09:59:59Z")), -8);
        assert_eq!(pacific_offset_hours(utc("2026-03-08T10:00:00Z")), -7);
        assert_eq!(pacific_offset_hours(utc("2026-10-03T12:00:00Z")), -7);
        assert_eq!(pacific_offset_hours(utc("2026-11-01T08:59:59Z")), -7);
        assert_eq!(pacific_offset_hours(utc("2026-11-01T09:00:00Z")), -8);
        assert_eq!(pacific_offset_hours(utc("2026-01-15T00:00:00Z")), -8);
    }

    #[test]
    fn pacific_day_spans_local_midnights() {
        // 2026-10-03 12:00 UTC is 05:00 PDT on Oct 3.
        let (start, end) = pacific_day(utc("2026-10-03T12:00:00Z"));
        assert_eq!(start, utc("2026-10-03T07:00:00Z"));
        assert_eq!(end, utc("2026-10-04T07:00:00Z"));
        // 03:00 UTC on Oct 3 is still Oct 2 in Pacific time.
        let (start, _) = pacific_day(utc("2026-10-03T03:00:00Z"));
        assert_eq!(start, utc("2026-10-02T07:00:00Z"));
        // In winter, midnight PST is 08:00 UTC.
        let (start, end) = pacific_day(utc("2026-01-15T20:00:00Z"));
        assert_eq!(start, utc("2026-01-15T08:00:00Z"));
        assert_eq!(end, utc("2026-01-16T08:00:00Z"));
        // The DST-start day is 23 hours long, the DST-end day 25.
        let (start, end) = pacific_day(utc("2026-03-08T20:00:00Z"));
        assert_eq!(end - start, Duration::hours(23));
        let (start, end) = pacific_day(utc("2026-11-01T20:00:00Z"));
        assert_eq!(end - start, Duration::hours(25));
    }

    fn msg(id: &str, model: &str, ts: &str, total: u64) -> serde_json::Value {
        serde_json::json!({
            "id": id, "type": "gemini", "model": model, "timestamp": ts,
            "content": [{"text": "secret conversation text"}],
            "tokens": {"input": 1, "output": 1, "cached": 0, "thoughts": 0, "tool": 0, "total": total},
        })
    }

    fn write_session(root: &Path, project: &str, name: &str, lines: &[serde_json::Value]) {
        let dir = root.join(project).join("chats");
        std::fs::create_dir_all(&dir).unwrap();
        let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(dir.join(name), body).unwrap();
    }

    // The file mtimes are "now" (the wall clock), so the day range used here
    // must contain the real present for the mtime filter to admit them. The
    // message timestamps are what decide what counts.
    fn today() -> (DateTime<Utc>, DateTime<Utc>) {
        let now = Utc::now();
        (now - Duration::hours(12), now + Duration::hours(12))
    }

    fn ts(offset_hours: i64) -> String {
        (Utc::now() + Duration::hours(offset_hours)).to_rfc3339()
    }

    #[test]
    fn counts_each_response_once_across_rewrites_and_set_patches() {
        let td = TempDir::new().unwrap();
        let (start, end) = today();
        write_session(
            td.path(),
            "proj",
            "session-a.jsonl",
            &[
                serde_json::json!({"sessionId": "s", "kind": "main"}),
                serde_json::json!({"id": "u1", "type": "user", "timestamp": ts(0), "content": []}),
                msg("m1", "gemini-2.5-flash", &ts(0), 100),
                // Rewritten while streaming: counted once, with the last total.
                msg("m1", "gemini-2.5-flash", &ts(0), 150),
                serde_json::json!({"$set": {"lastUpdated": ts(0)}}),
                serde_json::json!({"$set": {"messages": [
                    msg("m1", "gemini-2.5-flash", &ts(0), 150),
                    msg("m2", "gemini-2.5-pro", &ts(0), 1000),
                ]}}),
                serde_json::json!({"id": "e1", "type": "error", "timestamp": ts(0)}),
            ],
        );
        let totals = tally_dir(td.path(), start, end).unwrap();
        assert_eq!(totals["gemini-2.5-flash"], (1, 150));
        assert_eq!(totals["gemini-2.5-pro"], (1, 1000));
        assert_eq!(totals.len(), 2);
    }

    #[test]
    fn responses_outside_the_day_are_not_counted() {
        let td = TempDir::new().unwrap();
        let (start, end) = today();
        write_session(
            td.path(),
            "proj",
            "session-b.jsonl",
            &[
                msg("old", "gemini-2.5-flash", &ts(-20), 10),
                msg("new", "gemini-2.5-flash", &ts(0), 10),
                msg("future", "gemini-2.5-flash", &ts(20), 10),
            ],
        );
        let totals = tally_dir(td.path(), start, end).unwrap();
        assert_eq!(totals["gemini-2.5-flash"], (1, 10));
    }

    #[test]
    fn reads_legacy_json_sessions_and_several_projects() {
        let td = TempDir::new().unwrap();
        let (start, end) = today();
        let dir = td.path().join("old").join("chats");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("session-legacy.json"),
            serde_json::json!({"sessionId": "x", "messages": [msg("l1", "gemini-2.5-pro", &ts(0), 7)]})
                .to_string(),
        )
        .unwrap();
        write_session(
            td.path(),
            "new",
            "session-c.jsonl",
            &[msg("n1", "gemini-2.5-pro", &ts(0), 3)],
        );
        // Not a session file: ignored.
        std::fs::write(
            dir.join("notes.jsonl"),
            msg("x", "gemini-2.5-pro", &ts(0), 99).to_string(),
        )
        .unwrap();
        let totals = tally_dir(td.path(), start, end).unwrap();
        assert_eq!(totals["gemini-2.5-pro"], (2, 10));
    }

    #[test]
    fn a_log_untouched_since_before_the_day_is_not_opened() {
        let td = TempDir::new().unwrap();
        write_session(
            td.path(),
            "proj",
            "session-d.jsonl",
            &[msg("m", "gemini-2.5-pro", &ts(0), 1)],
        );
        // A day that starts after the file was written excludes it by mtime.
        let start = Utc::now() + Duration::hours(1);
        let totals = tally_dir(td.path(), start, start + Duration::days(1)).unwrap();
        assert!(totals.is_empty());
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let td = TempDir::new().unwrap();
        let (start, end) = today();
        let dir = td.path().join("proj").join("chats");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("session-e.jsonl"),
            format!(
                "{{not json\n{}\n{{\"id\":\"z\",\"type\":\"gemini\",\"model\":\"m\",\"timestamp\":\"yesterday\",\"tokens\":{{}}}}\n",
                msg("ok", "gemini-2.5-flash", &ts(0), 5)
            ),
        )
        .unwrap();
        let totals = tally_dir(td.path(), start, end).unwrap();
        assert_eq!(totals.len(), 1);
        assert_eq!(totals["gemini-2.5-flash"], (1, 5));
    }

    #[test]
    fn a_missing_logs_dir_says_to_run_the_cli() {
        let td = TempDir::new().unwrap();
        let (start, end) = today();
        let err = tally_dir(&td.path().join("nope"), start, end).unwrap_err();
        assert!(matches!(err, AppError::Credentials(ref m) if m.contains("run `gemini`")));
        assert!(!has_session_logs(&td.path().join("nope")));
    }

    #[test]
    fn has_session_logs_finds_a_session_file() {
        let td = TempDir::new().unwrap();
        assert!(!has_session_logs(td.path()));
        write_session(td.path(), "p", "session-f.jsonl", &[]);
        assert!(has_session_logs(td.path()));
    }

    #[test]
    fn snapshot_applies_limits_and_keeps_unlimited_models_as_counts() {
        let reset = utc("2026-10-04T07:00:00Z");
        let mut totals = BTreeMap::new();
        totals.insert("gemini-2.5-flash".to_string(), (125, 9_000));
        totals.insert("gemini-2.5-flash-lite".to_string(), (3, 300));
        let mut limits = BTreeMap::new();
        limits.insert("gemini-2.5-flash".to_string(), 250);
        limits.insert("gemini-2.5-pro".to_string(), 100);
        let snap = to_snapshot(totals, &limits, reset);
        assert_eq!(snap.source, GeminiSource::Local);
        assert_eq!(snap.plan, "API key");
        let order: Vec<&str> = snap.models.iter().map(|m| m.model.as_str()).collect();
        assert_eq!(
            order,
            [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite"
            ]
        );

        let pro = &snap.models[0];
        assert_eq!(
            (pro.requests, pro.utilization_pct, pro.remaining),
            (Some(0), Some(0), Some(100))
        );
        let flash = &snap.models[1];
        assert_eq!(flash.utilization_pct, Some(50));
        assert_eq!(flash.remaining, Some(125));
        assert_eq!(flash.tokens, Some(9_000));
        assert_eq!(flash.resets_at, Some(reset));
        let lite = &snap.models[2];
        assert_eq!(
            (lite.utilization_pct, lite.limit, lite.requests),
            (None, None, Some(3))
        );

        assert_eq!(snap.worst_pct(), Some(50));
        assert_eq!(snap.total_requests(), 128);
    }

    #[test]
    fn a_zero_limit_is_treated_as_unset_and_overuse_is_not_clamped_to_100() {
        let mut totals = BTreeMap::new();
        totals.insert("a".to_string(), (30, 0));
        totals.insert("b".to_string(), (5, 0));
        let mut limits = BTreeMap::new();
        limits.insert("a".to_string(), 20);
        limits.insert("b".to_string(), 0);
        let snap = to_snapshot(totals, &limits, Utc::now());
        let a = snap.models.iter().find(|m| m.model == "a").unwrap();
        assert_eq!((a.utilization_pct, a.remaining), (Some(150), Some(0)));
        let b = snap.models.iter().find(|m| m.model == "b").unwrap();
        assert_eq!(b.utilization_pct, None);
    }
}
