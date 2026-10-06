//! Wire shapes for the two Cloud Code calls the Gemini CLI makes:
//! `loadCodeAssist` (which project and tier the signed-in user has) and
//! `retrieveUserQuota` (one bucket per model). Both are read leniently — the
//! Gemini CLI itself ignores a bucket without `modelId` or `remainingFraction`,
//! and so does this parser — but a response with no usable bucket at all is an
//! error rather than an empty panel.

use chrono::{DateTime, Utc};
use serde::Deserialize;

use crate::error::{AppError, Result};
use crate::usage::{GeminiModelQuota, GeminiSnapshot};

/// What `loadCodeAssist` says about the user. Fields the CLI ignores for this
/// purpose (allowed tiers, privacy notices) are ignored here too.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadCodeAssistResponse {
    pub current_tier: Option<Tier>,
    pub paid_tier: Option<Tier>,
    pub cloudaicompanion_project: Option<ProjectRef>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Tier {
    pub id: Option<String>,
    pub name: Option<String>,
}

/// The project arrives as a bare id; an `{ "id": … }` object is accepted too,
/// the shape `onboardUser`'s response uses for the same value.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ProjectRef {
    Id(String),
    Object { id: Option<String> },
}

impl ProjectRef {
    fn id(&self) -> Option<&str> {
        match self {
            ProjectRef::Id(id) => Some(id.as_str()),
            ProjectRef::Object { id } => id.as_deref(),
        }
        .map(str::trim)
        .filter(|id| !id.is_empty())
    }
}

impl LoadCodeAssistResponse {
    /// The project to ask quota for, or why there is none. A user with no
    /// `currentTier` has never finished the CLI's first-run onboarding; this
    /// program never onboards anyone on their behalf.
    pub fn project(&self, configured: Option<&str>) -> Result<String> {
        if self.current_tier.is_none() {
            return Err(AppError::Credentials(
                "this Google account has not finished Gemini CLI setup; run `gemini` once to \
                 complete onboarding"
                    .into(),
            ));
        }
        if let Some(id) = self
            .cloudaicompanion_project
            .as_ref()
            .and_then(ProjectRef::id)
        {
            return Ok(id.to_string());
        }
        configured.map(str::to_string).ok_or_else(|| {
            AppError::Credentials(
                "Gemini Code Assist needs a Google Cloud project for this account; set \
                 [gemini] project (or GOOGLE_CLOUD_PROJECT) to the project the Gemini CLI uses"
                    .into(),
            )
        })
    }

    /// `paidTier` beats `currentTier`, name beats id — the CLI's own order —
    /// except that a recognised one-word tier (Ultra/Pro/Free) beats a long
    /// marketing name: the free tier's name is "Gemini Code Assist for
    /// individuals", while its id says `free-tier`.
    pub fn plan(&self) -> String {
        let candidates: Vec<String> = [self.paid_tier.as_ref(), self.current_tier.as_ref()]
            .into_iter()
            .flatten()
            .flat_map(|tier| [tier.name.as_deref(), tier.id.as_deref()])
            .flatten()
            .map(crate::antigravity::cloud::format_plan)
            .filter(|plan| !plan.is_empty())
            .collect();
        candidates
            .iter()
            .find(|plan| matches!(plan.as_str(), "Ultra" | "Pro" | "Free"))
            .or_else(|| candidates.first())
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct QuotaResponse {
    #[serde(default)]
    pub buckets: Vec<Bucket>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bucket {
    pub model_id: Option<String>,
    pub remaining_fraction: Option<f64>,
    /// int64 on the wire, so proto3 JSON sends it as a string.
    pub remaining_amount: Option<serde_json::Value>,
    pub reset_time: Option<String>,
    pub token_type: Option<String>,
}

/// Some Cloud Code channels wrap a body as `{ "response": … }`; accept either.
pub fn unwrap_response(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(mut map) if map.contains_key("response") => {
            map.remove("response").unwrap_or_default()
        }
        other => other,
    }
}

fn amount(value: &serde_json::Value) -> Option<u64> {
    match value {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn bucket_to_quota(bucket: &Bucket) -> Option<GeminiModelQuota> {
    let model = bucket.model_id.as_deref()?.trim();
    let fraction = bucket.remaining_fraction?;
    if model.is_empty() || !fraction.is_finite() {
        return None;
    }
    // A request count is the only unit this renderer labels; a bucket in some
    // other unit would be shown as "requests" wrongly, so its counts are
    // dropped and only the percentage survives.
    let counts_requests = bucket
        .token_type
        .as_deref()
        .is_none_or(|t| t.eq_ignore_ascii_case("REQUESTS"));
    let fraction = fraction.clamp(0.0, 1.0);
    let utilization_pct = ((1.0 - fraction) * 100.0).round() as i32;
    let remaining = bucket
        .remaining_amount
        .as_ref()
        .and_then(amount)
        .filter(|_| counts_requests);
    let limit = remaining
        .filter(|_| fraction > 0.0)
        .map(|r| (r as f64 / fraction).round() as u64)
        .filter(|limit| *limit > 0);
    let resets_at = bucket
        .reset_time
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s.trim()).ok())
        .map(|dt| dt.with_timezone(&Utc));
    Some(GeminiModelQuota {
        model: model.to_string(),
        utilization_pct: Some(utilization_pct),
        remaining,
        limit,
        requests: None,
        tokens: None,
        resets_at,
    })
}

/// Project a quota response onto a snapshot. One row per model: if a model
/// arrives in more than one bucket, the most-consumed one is kept, since that
/// is the one that will stop the user first.
pub fn to_snapshot(resp: &QuotaResponse, plan: String) -> Result<GeminiSnapshot> {
    let mut models: Vec<GeminiModelQuota> = Vec::new();
    for quota in resp.buckets.iter().filter_map(bucket_to_quota) {
        match models.iter_mut().find(|m| m.model == quota.model) {
            Some(existing) if quota.utilization_pct > existing.utilization_pct => {
                *existing = quota;
            }
            Some(_) => {}
            None => models.push(quota),
        }
    }
    if models.is_empty() {
        return Err(AppError::Schema(
            "Gemini quota response has no model buckets".into(),
        ));
    }
    let mut snap = GeminiSnapshot {
        plan,
        source: crate::usage::GeminiSource::Google,
        models,
    };
    snap.sort();
    Ok(snap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quota(json: serde_json::Value) -> QuotaResponse {
        serde_json::from_value(unwrap_response(json)).unwrap()
    }

    #[test]
    fn projects_the_cli_bucket_shape() {
        let resp = quota(serde_json::json!({
            "buckets": [
                {"modelId": "gemini-2.5-flash", "remainingFraction": 0.9,
                 "remainingAmount": "1350", "resetTime": "2026-10-04T07:00:00Z",
                 "tokenType": "REQUESTS"},
                {"modelId": "gemini-2.5-pro", "remainingFraction": 0.25,
                 "remainingAmount": "25", "resetTime": "2026-10-04T07:00:00Z",
                 "tokenType": "REQUESTS"},
            ]
        }));
        let snap = to_snapshot(&resp, "Free".into()).unwrap();
        assert_eq!(snap.plan, "Free");
        assert_eq!(snap.models[0].model, "gemini-2.5-pro");
        assert_eq!(snap.models[0].utilization_pct, Some(75));
        assert_eq!(snap.models[0].remaining, Some(25));
        assert_eq!(snap.models[0].limit, Some(100));
        assert_eq!(snap.models[1].model, "gemini-2.5-flash");
        assert_eq!(snap.models[1].limit, Some(1500));
        assert_eq!(
            snap.models[1].resets_at.unwrap().to_rfc3339(),
            "2026-10-04T07:00:00+00:00"
        );
        assert_eq!(snap.worst_pct(), Some(75));
    }

    #[test]
    fn the_response_wrapper_is_unwrapped() {
        let resp = quota(serde_json::json!({"response": {"buckets": [
            {"modelId": "gemini-2.5-pro", "remainingFraction": 1.0}
        ]}}));
        let snap = to_snapshot(&resp, String::new()).unwrap();
        assert_eq!(snap.models[0].utilization_pct, Some(0));
        assert_eq!(snap.models[0].remaining, None);
        assert_eq!(snap.models[0].limit, None);
    }

    #[test]
    fn unusable_buckets_are_skipped_and_none_left_is_an_error() {
        let resp = quota(serde_json::json!({"buckets": [
            {"remainingFraction": 0.5},
            {"modelId": "gemini-2.5-pro"},
            {"modelId": "  ", "remainingFraction": 0.5},
        ]}));
        assert!(matches!(
            to_snapshot(&resp, String::new()),
            Err(AppError::Schema(_))
        ));
        assert!(to_snapshot(&quota(serde_json::json!({})), String::new()).is_err());
    }

    #[test]
    fn duplicate_models_keep_the_most_consumed_bucket() {
        let resp = quota(serde_json::json!({"buckets": [
            {"modelId": "gemini-2.5-pro", "remainingFraction": 0.8},
            {"modelId": "gemini-2.5-pro", "remainingFraction": 0.1},
            {"modelId": "gemini-2.5-pro", "remainingFraction": 0.5},
        ]}));
        let snap = to_snapshot(&resp, String::new()).unwrap();
        assert_eq!(snap.models.len(), 1);
        assert_eq!(snap.models[0].utilization_pct, Some(90));
    }

    #[test]
    fn exhausted_bucket_has_no_derived_limit_and_fractions_are_clamped() {
        let resp = quota(serde_json::json!({"buckets": [
            {"modelId": "a-pro", "remainingFraction": 0.0, "remainingAmount": "0"},
            {"modelId": "b-flash", "remainingFraction": 1.7},
            {"modelId": "c-flash-lite", "remainingFraction": -0.2},
        ]}));
        let snap = to_snapshot(&resp, String::new()).unwrap();
        assert_eq!(snap.models[0].utilization_pct, Some(100));
        assert_eq!(snap.models[0].remaining, Some(0));
        assert_eq!(snap.models[0].limit, None);
        assert_eq!(snap.models[1].utilization_pct, Some(0));
        assert_eq!(snap.models[2].utilization_pct, Some(100));
    }

    #[test]
    fn non_request_buckets_keep_only_the_percentage() {
        let resp = quota(serde_json::json!({"buckets": [
            {"modelId": "gemini-2.5-pro", "remainingFraction": 0.5,
             "remainingAmount": "500000", "tokenType": "TOKENS"},
        ]}));
        let snap = to_snapshot(&resp, String::new()).unwrap();
        assert_eq!(snap.models[0].utilization_pct, Some(50));
        assert_eq!(snap.models[0].remaining, None);
    }

    #[test]
    fn models_sort_pro_flash_lite_then_other() {
        let resp = quota(serde_json::json!({"buckets": [
            {"modelId": "gemma-x", "remainingFraction": 1.0},
            {"modelId": "gemini-2.5-flash-lite", "remainingFraction": 1.0},
            {"modelId": "gemini-3-pro-preview", "remainingFraction": 1.0},
            {"modelId": "gemini-2.5-flash", "remainingFraction": 1.0},
            {"modelId": "gemini-2.5-pro", "remainingFraction": 1.0},
        ]}));
        let snap = to_snapshot(&resp, String::new()).unwrap();
        let order: Vec<&str> = snap.models.iter().map(|m| m.model.as_str()).collect();
        assert_eq!(
            order,
            [
                "gemini-2.5-pro",
                "gemini-3-pro-preview",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemma-x"
            ]
        );
    }

    fn load(json: serde_json::Value) -> LoadCodeAssistResponse {
        serde_json::from_value(unwrap_response(json)).unwrap()
    }

    #[test]
    fn project_comes_from_load_code_assist_then_config() {
        let found = load(serde_json::json!({
            "currentTier": {"id": "free-tier", "name": "Gemini Code Assist for individuals"},
            "cloudaicompanionProject": "proj-123",
        }));
        assert_eq!(found.project(Some("ignored")).unwrap(), "proj-123");

        let object = load(serde_json::json!({
            "currentTier": {"id": "standard-tier"},
            "cloudaicompanionProject": {"id": "proj-obj"},
        }));
        assert_eq!(object.project(None).unwrap(), "proj-obj");

        let bare = load(serde_json::json!({"currentTier": {"id": "standard-tier"}}));
        assert_eq!(bare.project(Some("mine")).unwrap(), "mine");
        assert!(
            matches!(bare.project(None), Err(AppError::Credentials(m)) if m.contains("GOOGLE_CLOUD_PROJECT"))
        );
    }

    #[test]
    fn a_user_who_never_onboarded_is_told_to_run_the_cli() {
        let fresh = load(serde_json::json!({"allowedTiers": [{"id": "free-tier"}]}));
        assert!(
            matches!(fresh.project(Some("p")), Err(AppError::Credentials(m)) if m.contains("run `gemini`"))
        );
    }

    #[test]
    fn plan_prefers_paid_tier_and_names() {
        let paid = load(serde_json::json!({
            "currentTier": {"id": "free-tier", "name": "Free"},
            "paidTier": {"id": "g1-pro-tier", "name": "Google AI Pro"},
        }));
        assert_eq!(paid.plan(), "Pro");
        let id_only = load(serde_json::json!({"currentTier": {"id": "free-tier"}}));
        assert_eq!(id_only.plan(), "Free");
        let marketing = load(serde_json::json!({
            "currentTier": {"id": "free-tier", "name": "Gemini Code Assist for individuals"},
        }));
        assert_eq!(marketing.plan(), "Free");
        let standard = load(serde_json::json!({"currentTier": {"id": "standard-tier"}}));
        assert_eq!(standard.plan(), "Standard Tier");
        assert_eq!(load(serde_json::json!({})).plan(), "");
    }
}
