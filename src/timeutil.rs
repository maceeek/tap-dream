//! Local-time helpers: splitting footage into "nights", capture windows,
//! and the human labels burned into the video.

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use chrono_tz::Tz;

/// Time zone wrapper: either a named IANA zone or the system local zone.
#[derive(Clone, Debug)]
pub enum Zone {
    Named(Tz),
    Local,
}

impl Zone {
    pub fn parse(name: &str) -> Result<Self> {
        if name.trim().is_empty() {
            return Ok(Zone::Local);
        }
        let tz: Tz = name.parse().map_err(|e| anyhow!("unknown time zone {name:?}: {e}"))?;
        Ok(Zone::Named(tz))
    }

    pub fn to_local(&self, ts: i64) -> NaiveDateTime {
        let utc = DateTime::<Utc>::from_timestamp(ts, 0).unwrap_or_default();
        match self {
            Zone::Named(tz) => utc.with_timezone(tz).naive_local(),
            Zone::Local => utc.with_timezone(&chrono::Local).naive_local(),
        }
    }

    /// Local wall time -> unix timestamp. Non-existent (DST gap) times are shifted forward,
    /// ambiguous ones take the earlier instant.
    pub fn timestamp_of(&self, dt: NaiveDateTime) -> i64 {
        fn resolve<T: TimeZone>(tz: &T, dt: NaiveDateTime) -> i64 {
            let mut probe = dt;
            for _ in 0..4 {
                if let Some(t) = tz.from_local_datetime(&probe).earliest() {
                    return t.timestamp();
                }
                probe += Duration::minutes(30);
            }
            tz.from_utc_datetime(&dt).timestamp()
        }
        match self {
            Zone::Named(tz) => resolve(tz, dt),
            Zone::Local => resolve(&chrono::Local, dt),
        }
    }
}

pub fn parse_hhmm(s: &str) -> Result<NaiveTime> {
    NaiveTime::parse_from_str(s.trim(), "%H:%M").with_context(|| format!("invalid time {s:?}, expected HH:MM"))
}

pub fn parse_date(s: &str) -> Result<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").with_context(|| format!("invalid date {s:?}, expected YYYY-MM-DD"))
}

/// Splits time into nights: night `D` = [D boundary, D+1 boundary) in local time.
#[derive(Clone, Debug)]
pub struct NightClock {
    pub zone: Zone,
    pub boundary: NaiveTime,
    /// Optional daily capture window (start, end) in local time; end <= start wraps over midnight.
    pub capture: Option<(NaiveTime, NaiveTime)>,
}

impl NightClock {
    pub fn new(zone: Zone, boundary: &str, capture: &str) -> Result<Self> {
        let boundary = parse_hhmm(boundary)?;
        let capture = if capture.trim().is_empty() {
            None
        } else {
            let (a, b) = capture.split_once('-').ok_or_else(|| anyhow!("capture_hours must look like 18:00-10:00"))?;
            let (a, b) = (parse_hhmm(a)?, parse_hhmm(b)?);
            if a == b {
                bail!("capture_hours start and end are equal");
            }
            Some((a, b))
        };
        Ok(Self { zone, boundary, capture })
    }

    /// The night a timestamp belongs to.
    pub fn night_of(&self, ts: i64) -> NaiveDate {
        let local = self.zone.to_local(ts);
        if local.time() >= self.boundary {
            local.date()
        } else {
            local.date() - Duration::days(1)
        }
    }

    /// [start, end) of a night as unix timestamps.
    pub fn night_range(&self, night: NaiveDate) -> (i64, i64) {
        let start = self.zone.timestamp_of(night.and_time(self.boundary));
        let end = self.zone.timestamp_of((night + Duration::days(1)).and_time(self.boundary));
        (start, end)
    }

    /// The sub-intervals of a night that fall inside the capture window.
    pub fn capture_intervals(&self, night: NaiveDate) -> Vec<(i64, i64)> {
        let (ns, ne) = self.night_range(night);
        let Some((a, b)) = self.capture else {
            return vec![(ns, ne)];
        };
        let mut out = Vec::new();
        for d in -1..=1 {
            let day = night + Duration::days(d);
            let start = self.zone.timestamp_of(day.and_time(a));
            let end_day = if b > a { day } else { day + Duration::days(1) };
            let end = self.zone.timestamp_of(end_day.and_time(b));
            let (s, e) = (start.max(ns), end.min(ne));
            if s < e {
                out.push((s, e));
            }
        }
        out.sort();
        out
    }

    #[cfg(test)]
    pub fn in_capture(&self, ts: i64) -> bool {
        let Some((a, b)) = self.capture else { return true };
        let t = self.zone.to_local(ts).time();
        if b > a {
            t >= a && t < b
        } else {
            t >= a || t < b
        }
    }

    /// Nights overlapped by [start, end).
    pub fn nights_overlapping(&self, start: i64, end: i64) -> Vec<NaiveDate> {
        let first = self.night_of(start);
        let last = self.night_of((end - 1).max(start));
        let mut out = Vec::new();
        let mut d = first;
        while d <= last {
            out.push(d);
            d += Duration::days(1);
        }
        out
    }
}

pub fn now_ts() -> i64 {
    Utc::now().timestamp()
}

pub fn fmt_ts(zone: &Zone, ts: i64) -> String {
    zone.to_local(ts).format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Calendar difference in (years, months, days), the way people count a baby's age.
pub fn age_parts(birth: NaiveDate, on: NaiveDate) -> (i32, i32, i32) {
    // Largest whole number of months that still fits (month ends are clamped:
    // Jan 31 + 1 month = Feb 28), the remainder are days.
    let mut months = (on.year() - birth.year()) * 12 + on.month() as i32 - birth.month() as i32;
    while months >= 0 {
        if let Some(anchor) = birth.checked_add_months(chrono::Months::new(months as u32)) {
            if anchor <= on {
                return (months / 12, months % 12, (on - anchor).num_days() as i32);
            }
        }
        months -= 1;
    }
    (0, 0, 0)
}

fn cs_plural(n: i32, one: &str, few: &str, many: &str) -> String {
    let w = match n {
        1 => one,
        2..=4 => few,
        _ => many,
    };
    format!("{n} {w}")
}

fn en_plural(n: i32, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

pub fn age_label(birth: NaiveDate, on: NaiveDate, lang: &str) -> String {
    let (y, m, d) = age_parts(birth, on);
    let mut parts = Vec::new();
    let cs = lang != "en";
    if y > 0 {
        parts.push(if cs { cs_plural(y, "rok", "roky", "let") } else { en_plural(y, "year", "years") });
    }
    if m > 0 {
        parts.push(if cs { cs_plural(m, "měsíc", "měsíce", "měsíců") } else { en_plural(m, "month", "months") });
    }
    if d > 0 || parts.is_empty() {
        parts.push(if cs { cs_plural(d, "den", "dny", "dní") } else { en_plural(d, "day", "days") });
    }
    parts.join(" ")
}

pub fn date_label(date: NaiveDate, lang: &str) -> String {
    if lang == "en" {
        date.format("%Y-%m-%d").to_string()
    } else {
        format!("{}. {}. {}", date.day(), date.month(), date.year())
    }
}

pub fn weekday_label(date: NaiveDate, lang: &str) -> &'static str {
    let i = date.weekday().num_days_from_monday() as usize;
    if lang == "en" {
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][i]
    } else {
        ["pondělí", "úterý", "středa", "čtvrtek", "pátek", "sobota", "neděle"][i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(capture: &str) -> NightClock {
        NightClock::new(Zone::parse("Europe/Prague").unwrap(), "12:00", capture).unwrap()
    }

    #[test]
    fn night_assignment() {
        let c = clock("");
        let d = NaiveDate::from_ymd_opt(2026, 10, 12).unwrap();
        let (s, e) = c.night_range(d);
        assert_eq!(e - s, 86400);
        assert_eq!(c.night_of(s), d);
        assert_eq!(c.night_of(e - 1), d);
        assert_eq!(c.night_of(e), d + Duration::days(1));
        // 2:00 local on the 13th belongs to the night of the 12th
        let t = c.zone.timestamp_of(NaiveDate::from_ymd_opt(2026, 10, 13).unwrap().and_hms_opt(2, 0, 0).unwrap());
        assert_eq!(c.night_of(t), d);
    }

    #[test]
    fn dst_night_is_25_hours() {
        let c = clock("");
        let d = NaiveDate::from_ymd_opt(2026, 10, 24).unwrap(); // DST ends 25 Oct 2026 in Europe
        let (s, e) = c.night_range(d);
        assert_eq!(e - s, 25 * 3600);
    }

    #[test]
    fn capture_window_wraps_midnight() {
        let c = clock("18:00-10:00");
        let d = NaiveDate::from_ymd_opt(2026, 10, 12).unwrap();
        let iv = c.capture_intervals(d);
        assert_eq!(iv.len(), 1);
        assert_eq!(iv[0].1 - iv[0].0, 16 * 3600);
        assert!(c.in_capture(iv[0].0));
        assert!(!c.in_capture(iv[0].1));
        let c2 = clock("20:00-23:00");
        assert_eq!(c2.capture_intervals(d).len(), 1);
        let c3 = clock("06:00-14:00"); // crosses the noon boundary -> two pieces
        assert_eq!(c3.capture_intervals(d).len(), 2);
    }

    #[test]
    fn ages() {
        let b = NaiveDate::from_ymd_opt(2026, 1, 31).unwrap();
        assert_eq!(age_parts(b, NaiveDate::from_ymd_opt(2026, 3, 1).unwrap()), (0, 1, 1));
        assert_eq!(age_label(b, b, "cs"), "0 dní");
        assert_eq!(age_label(b, NaiveDate::from_ymd_opt(2027, 4, 2).unwrap(), "cs"), "1 rok 2 měsíce 2 dny");
        assert_eq!(age_label(b, NaiveDate::from_ymd_opt(2031, 8, 5).unwrap(), "cs"), "5 let 6 měsíců 5 dní");
        assert_eq!(age_label(b, NaiveDate::from_ymd_opt(2026, 2, 1).unwrap(), "en"), "1 day");
    }
}
