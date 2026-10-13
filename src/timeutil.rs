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

/// What a piece of footage is used for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Part {
    /// Inside the night window.
    #[default]
    Night,
    /// Daytime: used only for naps (somebody is there long enough).
    Nap,
    /// Just outside the used footage: analysed only so the privacy window
    /// around a detection also covers frames near the edges.
    Context,
}

/// "08:00-10:00, 13:00-15:30" -> windows; an end <= start wraps over midnight.
pub fn parse_windows(s: &str) -> Result<Vec<(NaiveTime, NaiveTime)>> {
    let mut out = Vec::new();
    for w in s.split(',').map(str::trim).filter(|w| !w.is_empty()) {
        let (a, b) = w.split_once('-').ok_or_else(|| anyhow!("time window {w:?} must look like 18:00-08:00"))?;
        let (a, b) = (parse_hhmm(a)?, parse_hhmm(b)?);
        if a == b {
            bail!("time window {w:?} is empty");
        }
        out.push((a, b));
    }
    Ok(out)
}

/// Sorted union of intervals.
fn merge(mut v: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    v.retain(|(a, b)| a < b);
    v.sort();
    let mut out: Vec<(i64, i64)> = Vec::new();
    for (a, b) in v {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// `a` minus `b` (both sorted unions).
fn subtract(a: &[(i64, i64)], b: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for &(mut s, e) in a {
        for &(bs, be) in b {
            if be <= s || bs >= e {
                continue;
            }
            if bs > s {
                out.push((s, bs));
            }
            s = s.max(be);
            if s >= e {
                break;
            }
        }
        if s < e {
            out.push((s, e));
        }
    }
    out
}

/// Splits time into "days": day `D` = [D boundary, D+1 boundary) in local
/// time. With the default boundary of 17:00 that is the night from evening D
/// to morning D+1 plus the naps of D+1. Nights are counted by `D`.
#[derive(Clone, Debug)]
pub struct NightClock {
    pub zone: Zone,
    pub boundary: NaiveTime,
    /// Daily night windows; empty = the whole day counts as night.
    pub night: Vec<(NaiveTime, NaiveTime)>,
    /// Also use daytime footage (naps).
    pub naps: bool,
    /// Where to look for naps; empty = everything outside the night windows.
    pub nap: Vec<(NaiveTime, NaiveTime)>,
    /// Footage analysed (privacy only) before/after every used stretch.
    pub lead: i64,
    pub trail: i64,
}

impl NightClock {
    pub fn new(zone: Zone, boundary: &str, night_hours: &str) -> Result<Self> {
        Ok(Self {
            zone,
            boundary: parse_hhmm(boundary)?,
            night: parse_windows(night_hours)?,
            naps: false,
            nap: Vec::new(),
            lead: 0,
            trail: 0,
        })
    }

    pub fn with_naps(mut self, enabled: bool, nap_hours: &str) -> Result<Self> {
        self.naps = enabled;
        self.nap = parse_windows(nap_hours)?;
        Ok(self)
    }

    /// Privacy margins: a detection blocks `before` seconds back and `after`
    /// seconds ahead, so that much footage around every used stretch is checked.
    pub fn with_privacy_margins(mut self, before: f64, after: f64) -> Self {
        self.lead = after.ceil() as i64;
        self.trail = before.ceil() as i64;
        self
    }

    /// The day (night) a timestamp belongs to.
    pub fn night_of(&self, ts: i64) -> NaiveDate {
        let local = self.zone.to_local(ts);
        if local.time() >= self.boundary {
            local.date()
        } else {
            local.date() - Duration::days(1)
        }
    }

    /// [start, end) of a day as unix timestamps.
    pub fn night_range(&self, night: NaiveDate) -> (i64, i64) {
        let start = self.zone.timestamp_of(night.and_time(self.boundary));
        let end = self.zone.timestamp_of((night + Duration::days(1)).and_time(self.boundary));
        (start, end)
    }

    fn daily(&self, night: NaiveDate, windows: &[(NaiveTime, NaiveTime)]) -> Vec<(i64, i64)> {
        let (ns, ne) = self.night_range(night);
        if windows.is_empty() {
            return vec![(ns, ne)];
        }
        let mut out = Vec::new();
        for &(a, b) in windows {
            for d in -1..=1 {
                let day = night + Duration::days(d);
                let start = self.zone.timestamp_of(day.and_time(a));
                let end_day = if b > a { day } else { day + Duration::days(1) };
                let end = self.zone.timestamp_of(end_day.and_time(b));
                out.push((start.max(ns), end.min(ne)));
            }
        }
        merge(out)
    }

    /// The footage of a day that goes into the timelapse, by kind.
    pub fn parts(&self, night: NaiveDate) -> Vec<(i64, i64, Part)> {
        let nights = self.daily(night, &self.night);
        let mut out: Vec<(i64, i64, Part)> = nights.iter().map(|&(a, b)| (a, b, Part::Night)).collect();
        if self.naps && !self.night.is_empty() {
            let base = self.daily(night, &self.nap);
            out.extend(subtract(&base, &nights).into_iter().map(|(a, b)| (a, b, Part::Nap)));
        }
        out.sort_by_key(|p| p.0);
        out
    }

    /// Everything to decode for a day: the parts plus privacy context around them.
    pub fn plan(&self, night: NaiveDate) -> Vec<(i64, i64, Part)> {
        let mut out = self.parts(night);
        let used = merge(out.iter().map(|p| (p.0, p.1)).collect());
        let mut margins = Vec::new();
        for &(a, b) in &used {
            margins.push((a - self.lead, a));
            margins.push((b, b + self.trail));
        }
        let context = subtract(&merge(margins), &used);
        out.extend(context.into_iter().map(|(a, b)| (a, b, Part::Context)));
        out.sort_by_key(|p| p.0);
        out
    }

    /// The moment all footage a day needs has been recorded.
    pub fn plan_end(&self, night: NaiveDate) -> i64 {
        self.plan(night).iter().map(|p| p.1).max().unwrap_or_else(|| self.night_range(night).1)
    }

    /// Days whose plan may include footage from [start, end).
    pub fn nights_touching(&self, start: i64, end: i64) -> Vec<NaiveDate> {
        self.nights_overlapping(start - self.trail, end + self.lead)
    }

    /// Days overlapped by [start, end).
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

    fn clock(night: &str) -> NightClock {
        NightClock::new(Zone::parse("Europe/Prague").unwrap(), "12:00", night).unwrap()
    }

    fn at(c: &NightClock, d: u32, h: u32, m: u32) -> i64 {
        c.zone.timestamp_of(NaiveDate::from_ymd_opt(2026, 10, d).unwrap().and_hms_opt(h, m, 0).unwrap())
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
        assert_eq!(c.night_of(at(&c, 13, 2, 0)), d);
    }

    #[test]
    fn dst_night_is_25_hours() {
        let c = clock("");
        let d = NaiveDate::from_ymd_opt(2026, 10, 24).unwrap(); // DST ends 25 Oct 2026 in Europe
        let (s, e) = c.night_range(d);
        assert_eq!(e - s, 25 * 3600);
    }

    #[test]
    fn night_windows() {
        let d = NaiveDate::from_ymd_opt(2026, 10, 12).unwrap();
        let c = clock("18:00-10:00");
        let iv = c.parts(d);
        assert_eq!(iv, vec![(at(&c, 12, 18, 0), at(&c, 13, 10, 0), Part::Night)]);
        assert_eq!(clock("20:00-23:00").parts(d).len(), 1);
        assert_eq!(clock("06:00-14:00").parts(d).len(), 2); // crosses the noon boundary
        assert_eq!(clock("19:00-23:00, 01:00-06:00").parts(d).len(), 2);
        assert!(parse_windows("18:00").is_err());
    }

    #[test]
    fn naps_and_privacy_context() {
        let d = NaiveDate::from_ymd_opt(2026, 10, 12).unwrap();
        let z = Zone::parse("Europe/Prague").unwrap();
        let c = NightClock::new(z, "17:00", "19:00-07:00").unwrap().with_naps(true, "").unwrap().with_privacy_margins(120.0, 300.0);
        // 17:00-19:00 nap time, 19:00-07:00 night, 07:00-17:00 nap time
        let p = c.parts(d);
        assert_eq!(
            p,
            vec![
                (at(&c, 12, 17, 0), at(&c, 12, 19, 0), Part::Nap),
                (at(&c, 12, 19, 0), at(&c, 13, 7, 0), Part::Night),
                (at(&c, 13, 7, 0), at(&c, 13, 17, 0), Part::Nap),
            ]
        );
        // context only around the whole day (the parts touch each other)
        let plan = c.plan(d);
        assert_eq!(plan.first().unwrap(), &(at(&c, 12, 17, 0) - 300, at(&c, 12, 17, 0), Part::Context));
        assert_eq!(plan.last().unwrap(), &(at(&c, 13, 17, 0), at(&c, 13, 17, 0) + 120, Part::Context));
        assert_eq!(c.plan_end(d), at(&c, 13, 17, 2));
        // a recording just after the day belongs to the next day but is context for this one
        assert!(c.nights_touching(at(&c, 13, 17, 1), at(&c, 13, 17, 30)).contains(&d));
        // explicit nap windows
        let c2 = NightClock::new(Zone::parse("Europe/Prague").unwrap(), "17:00", "19:00-07:00")
            .unwrap()
            .with_naps(true, "09:00-11:00, 13:00-16:00")
            .unwrap()
            .with_privacy_margins(120.0, 300.0);
        let naps: Vec<_> = c2.parts(d).into_iter().filter(|p| p.2 == Part::Nap).collect();
        assert_eq!(naps, vec![(at(&c, 13, 9, 0), at(&c, 13, 11, 0), Part::Nap), (at(&c, 13, 13, 0), at(&c, 13, 16, 0), Part::Nap)]);
        let ctx = c2.plan(d).into_iter().filter(|p| p.2 == Part::Context).count();
        assert_eq!(ctx, 6); // before and after each of the three stretches
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
