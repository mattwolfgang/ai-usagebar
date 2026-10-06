//! Waybar renderer for a self-hosted Ollama server: today's token total on
//! the bar, the per-source and per-model split in the tooltip. A tally has no
//! limit, so it never changes severity — the class stays `low`.

use std::collections::HashMap;

use chrono::{DateTime, Local, Utc};

use crate::format::{placeholders, substitute, updated_at_hm};
use crate::pango::{color_span, escape};
use crate::theme::Theme;
use crate::tooltip::{Line as TooltipLine, render_bordered};
use crate::usage::{OllamaLocalSnapshot, OllamaLocalSource, compact_tokens};
use crate::vendor::{RenderOpts, VendorId, VendorOutcome};
use crate::waybar::{Class, WaybarOutput};

use super::fetch::FetchOutcome;

pub const DEFAULT_FORMAT: &str = "{olo_tokens}";

/// Nerd Font `nf-md-llama`.
const DEFAULT_ICON: &str = "󰳆";

/// "today" when the snapshot covers the local day containing `now`, else the
/// date — a stale payload carried over midnight is yesterday's total.
pub fn day_label(snap: &OllamaLocalSnapshot, now: DateTime<Utc>) -> String {
    if snap.day == now.with_timezone(&Local).date_naive() {
        "today".into()
    } else {
        snap.day.format("%b %-d").to_string()
    }
}

pub fn build_placeholders(
    snap: &OllamaLocalSnapshot,
    now: DateTime<Utc>,
) -> HashMap<&'static str, String> {
    placeholders(vec![
        ("icon", DEFAULT_ICON.to_string()),
        (
            "vendor_short",
            VendorId::OllamaLocal.short_name().to_string(),
        ),
        ("plan", snap.server.clone()),
        ("olo_tokens", compact_tokens(snap.total_tokens())),
        ("olo_tokens_exact", snap.total_tokens().to_string()),
        ("olo_requests", snap.total_requests().to_string()),
        (
            "olo_opencode_tokens",
            compact_tokens(snap.source_tokens(OllamaLocalSource::OpenCode)),
        ),
        (
            "olo_openwebui_tokens",
            compact_tokens(snap.source_tokens(OllamaLocalSource::OpenWebUi)),
        ),
        ("olo_day", day_label(snap, now)),
        ("olo_server", snap.server.clone()),
    ])
}

pub fn render(
    outcome: &VendorOutcome,
    snap: &OllamaLocalSnapshot,
    theme: &Theme,
    opts: &RenderOpts,
    now: DateTime<Utc>,
) -> WaybarOutput {
    let format = opts
        .format
        .clone()
        .unwrap_or_else(|| DEFAULT_FORMAT.to_string());
    let mut values = build_placeholders(snap, now);
    // The server label comes from config, but both sinks are Pango markup.
    for key in ["plan", "olo_server"] {
        if let Some(value) = values.get_mut(key) {
            *value = escape(value);
        }
    }

    let mut text = substitute(&format, &values);
    if outcome.stale {
        text.push_str(" ⏸");
    }
    let icon_prefix = match opts.icon.as_deref() {
        Some(ic) if !ic.is_empty() => format!("{ic} "),
        _ => String::new(),
    };
    let bar_text = color_span(&theme.fg, &format!("{icon_prefix}{text}"));

    let tooltip = if let Some(fmt) = opts.tooltip_format.as_deref() {
        substitute(fmt, &values)
    } else {
        render_tooltip(outcome, snap, theme, now)
    };

    WaybarOutput {
        text: bar_text,
        tooltip,
        class: Class::Low,
    }
}

fn render_tooltip(
    outcome: &VendorOutcome,
    snap: &OllamaLocalSnapshot,
    theme: &Theme,
    now: DateTime<Utc>,
) -> String {
    let blue = &theme.blue;
    let dim = &theme.dim;
    let fg = &theme.fg;

    let mut lines: Vec<TooltipLine> = Vec::new();
    lines.push(TooltipLine::Center(format!(
        "<span font_weight='bold' foreground='{blue}'>Ollama · {}</span>",
        escape(&snap.server)
    )));
    lines.push(TooltipLine::Sep);
    lines.push(TooltipLine::Body("".into()));
    lines.push(TooltipLine::Body(format!(
        " <span foreground='{fg}'>  󰆙  Tokens {}</span>",
        day_label(snap, now)
    )));
    lines.push(TooltipLine::Body(format!(
        "   <span font_weight='bold' foreground='{fg}'>{}</span> \
         <span foreground='{dim}'>({} replies)</span>",
        compact_tokens(snap.total_tokens()),
        snap.total_requests()
    )));
    for source in [OllamaLocalSource::OpenCode, OllamaLocalSource::OpenWebUi] {
        if snap.models.iter().any(|m| m.source == source) {
            lines.push(TooltipLine::Body(format!(
                "   <span foreground='{dim}'>{}: {}</span>",
                source.label(),
                compact_tokens(snap.source_tokens(source))
            )));
        }
    }

    let by_model = snap.by_model();
    if !by_model.is_empty() {
        lines.push(TooltipLine::Body("".into()));
        lines.push(TooltipLine::Body(format!(
            " <span foreground='{fg}'>  󰧑  Models</span>"
        )));
        for (model, tokens) in by_model.iter().take(5) {
            lines.push(TooltipLine::Body(format!(
                "   <span foreground='{dim}'>{}: {}</span>",
                escape(model),
                compact_tokens(*tokens)
            )));
        }
    }

    if let Some((code, msg)) = outcome.last_error.as_ref() {
        let (icon, ecolor) = if *code >= 500 {
            ("󰅚", theme.red.as_str())
        } else {
            ("󰀪", theme.orange.as_str())
        };
        let head = if *code == 0 {
            "Refresh failed".to_string()
        } else {
            format!("HTTP {code}")
        };
        lines.push(TooltipLine::Body("".into()));
        lines.push(TooltipLine::Sep);
        lines.push(TooltipLine::Body(format!(
            " <span foreground='{ecolor}'>  {icon}  {head}</span>"
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
        o.map(crate::usage::VendorSnapshot::OllamaLocal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::OllamaLocalModel;

    /// Fixed, and `snap().day` is derived from it in the same zone, so the
    /// "today" check holds whatever the machine's timezone is.
    fn now() -> DateTime<Utc> {
        use chrono::TimeZone;
        Utc.with_ymd_and_hms(2026, 10, 6, 17, 0, 0).unwrap()
    }

    fn snap() -> OllamaLocalSnapshot {
        OllamaLocalSnapshot {
            server: "192.168.1.50".into(),
            day: now().with_timezone(&Local).date_naive(),
            models: vec![
                OllamaLocalModel {
                    source: OllamaLocalSource::OpenCode,
                    model: "qwen3.6-35b-a3b-32k".into(),
                    requests: 13,
                    input_tokens: 52_130,
                    output_tokens: 6_347,
                },
                OllamaLocalModel {
                    source: OllamaLocalSource::OpenWebUi,
                    model: "qwen3.5:9b".into(),
                    requests: 2,
                    input_tokens: 900,
                    output_tokens: 100,
                },
            ],
        }
    }

    fn outcome(snap: OllamaLocalSnapshot) -> VendorOutcome {
        VendorOutcome {
            snapshot: crate::usage::VendorSnapshot::OllamaLocal(snap),
            stale: false,
            last_error: None,
            cache_age: Some(std::time::Duration::from_secs(5)),
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

    #[test]
    fn bar_shows_the_compact_total() {
        let s = snap();
        let out = render(&outcome(s.clone()), &s, &Theme::default(), &opts(), now());
        assert!(out.text.contains("59.5k"), "{}", out.text);
        assert_eq!(out.class, Class::Low);
    }

    #[test]
    fn tooltip_splits_by_source_and_model() {
        let s = snap();
        let out = render(&outcome(s.clone()), &s, &Theme::default(), &opts(), now());
        assert!(out.tooltip.contains("Ollama · 192.168.1.50"));
        assert!(out.tooltip.contains("Tokens today"));
        assert!(out.tooltip.contains("15 replies"));
        assert!(out.tooltip.contains("OpenCode: 58.5k"));
        assert!(out.tooltip.contains("Open WebUI: 1k"));
        assert!(out.tooltip.contains("qwen3.6-35b-a3b-32k: 58.5k"));
    }

    #[test]
    fn a_carried_over_total_names_its_day_and_pauses() {
        let mut s = snap();
        s.day = s.day.pred_opt().unwrap();
        let mut o = outcome(s.clone());
        o.stale = true;
        let out = render(&o, &s, &Theme::default(), &opts(), now());
        assert!(out.text.contains("⏸"));
        assert!(!out.tooltip.contains("Tokens today"));
        assert_eq!(
            build_placeholders(&s, now())["olo_day"],
            s.day.format("%b %-d").to_string()
        );
    }

    #[test]
    fn custom_formats_use_placeholders() {
        let s = snap();
        let mut o = opts();
        o.format = Some("{olo_tokens_exact}/{olo_requests}".into());
        o.tooltip_format = Some("{olo_opencode_tokens}+{olo_openwebui_tokens}".into());
        let out = render(&outcome(s.clone()), &s, &Theme::default(), &o, now());
        assert!(out.text.contains("59477/15"), "{}", out.text);
        assert_eq!(out.tooltip, "58.5k+1k");
    }

    #[test]
    fn the_server_label_is_inert_markup() {
        let mut s = snap();
        s.server = "<b>x</b>".into();
        let mut o = opts();
        o.format = Some("{plan}".into());
        let out = render(&outcome(s.clone()), &s, &Theme::default(), &o, now());
        assert!(!out.text.contains("<b>"));
    }

    #[test]
    fn fetch_outcome_conversion_preserves_metadata() {
        let fetch = FetchOutcome {
            snapshot: snap(),
            stale: true,
            last_error: Some((0, "down".into())),
            cache_age: None,
        };
        let vendor: VendorOutcome = fetch.into();
        assert!(matches!(
            vendor.snapshot,
            crate::usage::VendorSnapshot::OllamaLocal(_)
        ));
        assert!(vendor.stale);
    }
}
