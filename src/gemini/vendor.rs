//! Gemini CLI renderer — one row per model. With a percentage (Google's quota,
//! or a local tally against a configured daily limit) the bar shows the
//! most-consumed bucket, the one that will stop the user first. A local tally
//! with no limits has only counts, so the bar shows today's request count.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::countdown;
use crate::format::{placeholders, substitute, updated_at_hm};
use crate::pacing::PaceSeverity;
use crate::pango::{color_span, escape, severity_color, severity_for};
use crate::theme::Theme;
use crate::tooltip::{Line as TooltipLine, render_bordered};
use crate::usage::{GeminiModelQuota, GeminiSnapshot, GeminiSource};
use crate::vendor::{RenderOpts, VendorId, VendorOutcome};
use crate::waybar::{Class, WaybarOutput};

use super::fetch::FetchOutcome;

pub const DEFAULT_FORMAT: &str = "{gemini_pct}%";

/// The default when no model has a percentage: a local tally with no
/// configured limits has nothing to measure against, only a count.
pub const DEFAULT_COUNT_FORMAT: &str = "{gemini_requests} req";

/// Nerd Font `nf-md-google` — a Gemini mark is not in the common sets.
const DEFAULT_ICON: &str = "󰊭";

/// "45 of 100 left" when the bucket states counts; empty otherwise.
fn counts(m: &GeminiModelQuota) -> String {
    match (m.remaining, m.limit) {
        (Some(remaining), Some(limit)) => format!("{remaining} of {limit} left"),
        (Some(remaining), None) => format!("{remaining} left"),
        _ => String::new(),
    }
}

/// "12 requests · 34.5k tokens today" for a local tally row; empty otherwise.
pub fn tally(m: &GeminiModelQuota) -> String {
    let Some(requests) = m.requests else {
        return String::new();
    };
    let noun = if requests == 1 { "request" } else { "requests" };
    match m.tokens {
        Some(tokens) => format!("{requests} {noun} · {} tokens today", compact(tokens)),
        None => format!("{requests} {noun} today"),
    }
}

/// 950 → "950", 34_512 → "34.5k", 2_100_000 → "2.1M".
pub fn compact(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.1}k", n as f64 / 1_000.0),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

/// The first model of a family, skipping Flash-Lite when asking for Flash.
fn family<'a>(snap: &'a GeminiSnapshot, needle: &str) -> Option<&'a GeminiModelQuota> {
    snap.models.iter().find(|m| {
        m.model.contains(needle) && (needle != "flash" || !m.model.contains("flash-lite"))
    })
}

pub fn build_placeholders(
    snap: &GeminiSnapshot,
    now: DateTime<Utc>,
) -> HashMap<&'static str, String> {
    let pct_of = |m: &GeminiModelQuota| m.utilization_pct.map_or(String::new(), |p| p.to_string());
    let worst = snap.worst();
    let pct = worst.map_or(String::new(), pct_of);
    // Every model shares the local day's reset; Google's buckets each carry
    // their own, and the worst one's is the one that matters.
    let reset_of = worst.or(snap.models.first());
    let reset = reset_of.map_or(String::new(), |m| countdown::format(m.resets_at, now));
    let model = worst.map_or(String::new(), |m| m.model.clone());
    let family_pct = |needle| family(snap, needle).map_or(String::new(), pct_of);
    placeholders(vec![
        ("icon", DEFAULT_ICON.to_string()),
        ("vendor_short", VendorId::Gemini.short_name().to_string()),
        // Cross-vendor aliases: the worst bucket fills both generic slots, as
        // Kiro's single pool does.
        ("plan", snap.plan.clone()),
        ("session_pct", pct.clone()),
        ("session_reset", reset.clone()),
        ("weekly_pct", pct.clone()),
        ("weekly_reset", reset.clone()),
        // Gemini-specific placeholders.
        ("gemini_plan", snap.plan.clone()),
        ("gemini_pct", pct),
        ("gemini_model", model),
        ("gemini_reset", reset),
        ("gemini_pro_pct", family_pct("pro")),
        ("gemini_flash_pct", family_pct("flash")),
        ("gemini_requests", snap.total_requests().to_string()),
        ("gemini_tokens", compact(snap.total_tokens())),
    ])
}

/// A count with no limit is not a warning, so it grades Low.
pub fn severity(snap: &GeminiSnapshot) -> PaceSeverity {
    snap.worst_pct().map_or(PaceSeverity::Low, severity_for)
}

pub fn render(
    outcome: &VendorOutcome,
    snap: &GeminiSnapshot,
    theme: &Theme,
    opts: &RenderOpts,
    now: DateTime<Utc>,
) -> WaybarOutput {
    let class = Class::from(severity(snap));
    let format = opts.format.clone().unwrap_or_else(|| {
        if snap.worst_pct().is_some() {
            DEFAULT_FORMAT
        } else {
            DEFAULT_COUNT_FORMAT
        }
        .to_string()
    });
    let mut values = build_placeholders(snap, now);
    // Both sinks fed by this map are Pango markup, and the plan and model id
    // come from the API: escape them at the projection boundary.
    for key in ["plan", "gemini_plan", "gemini_model"] {
        if let Some(value) = values.get_mut(key) {
            *value = escape(value);
        }
    }

    let mut text = substitute(&format, &values);
    if outcome.stale {
        text.push_str(" ⏸");
    }

    let wrapper_color = severity_color(severity(snap), theme).to_string();
    let icon_prefix = match opts.icon.as_deref() {
        Some(ic) if !ic.is_empty() => format!("{ic} "),
        _ => String::new(),
    };
    let bar_text = color_span(&wrapper_color, &format!("{icon_prefix}{text}"));

    let tooltip = if let Some(fmt) = opts.tooltip_format.as_deref() {
        substitute(fmt, &values)
    } else {
        render_tooltip(outcome, snap, theme, now)
    };

    WaybarOutput {
        text: bar_text,
        tooltip,
        class,
    }
}

fn render_tooltip(
    outcome: &VendorOutcome,
    snap: &GeminiSnapshot,
    theme: &Theme,
    now: DateTime<Utc>,
) -> String {
    let blue = &theme.blue;
    let dim = &theme.dim;
    let fg = &theme.fg;

    let mut lines: Vec<TooltipLine> = Vec::new();
    let title = if snap.plan.is_empty() {
        "Gemini".to_string()
    } else {
        format!("Gemini {}", escape(&snap.plan))
    };
    lines.push(TooltipLine::Center(format!(
        "<span font_weight='bold' foreground='{blue}'>{title}</span>"
    )));
    lines.push(TooltipLine::Sep);

    if snap.models.is_empty() {
        lines.push(TooltipLine::Body("".into()));
        lines.push(TooltipLine::Body(format!(
            " <span foreground='{fg}'>  No requests yet today</span>"
        )));
    }
    for m in &snap.models {
        lines.push(TooltipLine::Body("".into()));
        lines.push(TooltipLine::Body(format!(
            " <span foreground='{fg}'>  󰧑  {}</span>",
            escape(&m.model)
        )));
        if let Some(pct) = m.utilization_pct {
            let color = severity_color(severity_for(pct), theme);
            let counts = counts(m);
            let counts = if counts.is_empty() {
                String::new()
            } else {
                format!(" <span foreground='{dim}'>({counts})</span>")
            };
            lines.push(TooltipLine::Body(format!(
                "   <span font_weight='bold' foreground='{color}'>{pct}%</span> used{counts}"
            )));
        }
        let tally = tally(m);
        if !tally.is_empty() {
            lines.push(TooltipLine::Body(format!(
                "   <span foreground='{fg}'>{tally}</span>"
            )));
        }
        lines.push(TooltipLine::Body(format!(
            "   <span foreground='{dim}'>󰃰  Resets {}</span>",
            escape(&countdown::format(m.resets_at, now))
        )));
    }

    if snap.source == GeminiSource::Local {
        lines.push(TooltipLine::Body("".into()));
        lines.push(TooltipLine::Body(format!(
            " <span foreground='{dim}'>  Counted from this machine's Gemini CLI logs</span>"
        )));
    }

    if let Some((code, msg)) = outcome.last_error.as_ref() {
        let (icon, ecolor) = if *code >= 500 {
            ("󰅚", theme.red.as_str())
        } else {
            ("󰀪", theme.orange.as_str())
        };
        let heading = if *code == 0 {
            "Error".to_string()
        } else {
            format!("HTTP {code}")
        };
        lines.push(TooltipLine::Body("".into()));
        lines.push(TooltipLine::Sep);
        lines.push(TooltipLine::Body(format!(
            " <span foreground='{ecolor}'>  {icon}  {heading}</span>"
        )));
        lines.push(TooltipLine::Body(format!(
            "     <span foreground='{dim}'>{}</span>",
            escape(msg)
        )));
    }

    let updated = updated_at_hm(now, outcome.cache_age);
    lines.push(TooltipLine::Body("".into()));
    lines.push(TooltipLine::Sep);
    lines.push(TooltipLine::Body(format!(
        " <span foreground='{dim}'>  󰅐  Updated {updated}</span>"
    )));

    render_bordered(&lines, theme)
}

impl From<FetchOutcome> for VendorOutcome {
    fn from(o: FetchOutcome) -> Self {
        o.map(crate::usage::VendorSnapshot::Gemini)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 3, 12, 0, 0).unwrap()
    }

    fn sample_snap() -> GeminiSnapshot {
        let reset = Some(now() + chrono::Duration::hours(19));
        GeminiSnapshot {
            plan: "Free".into(),
            source: GeminiSource::Google,
            models: vec![
                GeminiModelQuota {
                    model: "gemini-2.5-pro".into(),
                    utilization_pct: Some(80),
                    remaining: Some(20),
                    limit: Some(100),
                    requests: None,
                    tokens: None,
                    resets_at: reset,
                },
                GeminiModelQuota {
                    model: "gemini-2.5-flash".into(),
                    utilization_pct: Some(10),
                    remaining: None,
                    limit: None,
                    requests: None,
                    tokens: None,
                    resets_at: reset,
                },
                GeminiModelQuota {
                    model: "gemini-2.5-flash-lite".into(),
                    utilization_pct: Some(95),
                    remaining: Some(50),
                    limit: Some(1000),
                    requests: None,
                    tokens: None,
                    resets_at: reset,
                },
            ],
        }
    }

    fn sample_outcome(snap: GeminiSnapshot) -> VendorOutcome {
        VendorOutcome {
            snapshot: crate::usage::VendorSnapshot::Gemini(snap),
            stale: false,
            last_error: None,
            cache_age: Some(std::time::Duration::from_secs(10)),
        }
    }

    fn opts() -> RenderOpts {
        RenderOpts {
            format: None,
            tooltip_format: None,
            icon: None,
            pace_tolerance: 5,
            format_pace_color: false,
            tooltip_pace_pts: false,
        }
    }

    fn render_default(snap: &GeminiSnapshot) -> WaybarOutput {
        render(
            &sample_outcome(snap.clone()),
            snap,
            &Theme::default(),
            &opts(),
            now(),
        )
    }

    #[test]
    fn bar_shows_the_most_consumed_bucket() {
        let out = render_default(&sample_snap());
        assert!(out.text.contains("95%"), "text: {}", out.text);
    }

    #[test]
    fn severity_follows_the_worst_bucket() {
        assert_eq!(severity(&sample_snap()), PaceSeverity::Critical);
    }

    #[test]
    fn tooltip_lists_every_model_with_counts_and_reset() {
        let out = render_default(&sample_snap());
        assert!(out.tooltip.contains("Gemini Free"));
        for model in [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ] {
            assert!(out.tooltip.contains(model), "missing {model}");
        }
        assert!(out.tooltip.contains("20 of 100 left"));
        assert!(out.tooltip.contains("19h"));
    }

    #[test]
    fn tooltip_shows_a_credentials_error_without_an_http_code() {
        let snap = sample_snap();
        let mut outcome = sample_outcome(snap.clone());
        outcome.stale = true;
        outcome.last_error = Some((0, "run `gemini` to renew it".into()));
        let out = render(&outcome, &snap, &Theme::default(), &opts(), now());
        assert!(out.text.contains("⏸"));
        assert!(out.tooltip.contains("Error"));
        assert!(out.tooltip.contains("renew it"));
    }

    #[test]
    fn family_placeholders_skip_flash_lite_for_flash() {
        let values = build_placeholders(&sample_snap(), now());
        assert_eq!(values["gemini_pro_pct"], "80");
        assert_eq!(values["gemini_flash_pct"], "10");
        assert_eq!(values["gemini_model"], "gemini-2.5-flash-lite");
        assert_eq!(values["session_pct"], "95");
        assert_eq!(values["weekly_pct"], "95");
    }

    #[test]
    fn api_text_is_inert_in_custom_pango_formats() {
        let mut snap = sample_snap();
        snap.plan = "<b>plan</b>".into();
        snap.models[2].model = "<i>model</i>".into();
        let mut o = opts();
        o.format = Some("{gemini_model} {plan}".into());
        o.tooltip_format = Some("{gemini_plan}".into());
        let out = render(
            &sample_outcome(snap.clone()),
            &snap,
            &Theme::default(),
            &o,
            now(),
        );
        assert!(!out.text.contains("<i>") && !out.text.contains("<b>"));
        assert_eq!(out.tooltip, "&lt;b&gt;plan&lt;/b&gt;");
    }

    fn local_snap(limit: Option<u64>) -> GeminiSnapshot {
        let mut limits = std::collections::BTreeMap::new();
        if let Some(limit) = limit {
            limits.insert("gemini-2.5-flash".to_string(), limit);
        }
        let mut totals = std::collections::BTreeMap::new();
        totals.insert("gemini-2.5-flash".to_string(), (42, 34_512));
        totals.insert("gemini-2.5-pro".to_string(), (1, 900));
        crate::gemini::local::to_snapshot(totals, &limits, now() + chrono::Duration::hours(19))
    }

    #[test]
    fn local_tally_without_limits_shows_a_request_count_calmly() {
        let snap = local_snap(None);
        let out = render_default(&snap);
        assert!(out.text.contains("43 req"), "text: {}", out.text);
        assert_eq!(severity(&snap), PaceSeverity::Low);
        assert!(out.tooltip.contains("Gemini API key"));
        assert!(out.tooltip.contains("42 requests · 34.5k tokens today"));
        assert!(out.tooltip.contains("1 request · 900 tokens today"));
        assert!(out.tooltip.contains("this machine's Gemini CLI logs"));
        assert!(!out.tooltip.contains("% used"));
        let values = build_placeholders(&snap, now());
        assert_eq!(values["gemini_pct"], "");
        assert_eq!(values["gemini_tokens"], "35.4k");
        assert_eq!(values["session_reset"], "19h 00m");
    }

    #[test]
    fn local_tally_with_a_limit_shows_a_percentage() {
        let snap = local_snap(Some(50));
        let out = render_default(&snap);
        assert!(out.text.contains("84%"), "text: {}", out.text);
        assert_eq!(severity(&snap), PaceSeverity::High);
        assert!(out.tooltip.contains("8 of 50 left"));
    }

    #[test]
    fn an_empty_local_day_says_so() {
        let snap = crate::gemini::local::to_snapshot(
            Default::default(),
            &Default::default(),
            now() + chrono::Duration::hours(1),
        );
        let out = render_default(&snap);
        assert!(out.text.contains("0 req"));
        assert!(out.tooltip.contains("No requests yet today"));
    }

    #[test]
    fn compact_counts() {
        assert_eq!(compact(950), "950");
        assert_eq!(compact(34_512), "34.5k");
        assert_eq!(compact(2_100_000), "2.1M");
    }

    #[test]
    fn fetch_outcome_conversion_preserves_metadata() {
        let fetch = FetchOutcome {
            snapshot: sample_snap(),
            stale: true,
            last_error: Some((401, "bad".into())),
            cache_age: Some(std::time::Duration::from_secs(42)),
        };
        let vendor: VendorOutcome = fetch.into();
        assert!(matches!(
            vendor.snapshot,
            crate::usage::VendorSnapshot::Gemini(_)
        ));
        assert!(vendor.stale);
    }
}
