//! The index: which recordings exist on the camera, which were downloaded,
//! which nights were turned into timelapse segments and what the master file
//! looks like. Everything that must survive a crash lives here.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

pub struct Db {
    pub conn: Connection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecStatus {
    New,
    Downloaded,
    Failed,
    Skipped,
    Purged,
}

impl RecStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RecStatus::New => "new",
            RecStatus::Downloaded => "downloaded",
            RecStatus::Failed => "failed",
            RecStatus::Skipped => "skipped",
            RecStatus::Purged => "purged",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "downloaded" => RecStatus::Downloaded,
            "failed" => RecStatus::Failed,
            "skipped" => RecStatus::Skipped,
            "purged" => RecStatus::Purged,
            _ => RecStatus::New,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // row mirror; not every column is used by every command
pub struct Recording {
    pub id: i64,
    pub cam_start: i64,
    pub cam_end: i64,
    pub start_ts: i64,
    pub end_ts: i64,
    pub video_type: i64,
    pub status: RecStatus,
    pub file: Option<String>,
    pub duration: Option<f64>,
    pub attempts: i64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct NightStats {
    pub analyzed: i64,
    pub emitted: i64,
    pub skipped_private: i64,
    pub skipped_idle: i64,
    pub skipped_dark: i64,
    /// Frames from naps (included in `emitted`).
    pub nap_frames: i64,
    /// Frames of daytime visits too short to count as a nap.
    pub skipped_short: i64,
    pub best_ts: Option<f64>,
    pub best_score: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct NightRow {
    pub night: String,
    pub status: String,
    pub stats: NightStats,
    pub segment: Option<i64>,
    pub uncensored_segment: Option<i64>,
    /// Frames of the uncensored segment (0 = none).
    pub uncensored_frames: i64,
}

/// The timelapse exists in up to two versions built from the same selection:
/// the regular one with the privacy filter and an optional one without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Censored,
    Uncensored,
}

impl Variant {
    pub fn key(self) -> &'static str {
        match self {
            Variant::Censored => "censored",
            Variant::Uncensored => "uncensored",
        }
    }

    /// Column of `nights` that points at this variant's segment.
    fn night_column(self) -> &'static str {
        match self {
            Variant::Censored => "segment_id",
            Variant::Uncensored => "uncensored_segment_id",
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Segment {
    pub id: i64,
    pub night: String,
    pub file: String,
    pub frames: i64,
    pub duration: i64,
    pub appended: bool,
}

#[derive(Debug, Clone)]
pub struct FrameRow {
    pub idx: i64,
    pub ts: f64,
    pub motion: f32,
    pub private: f32,
    pub presence: f32,
    pub interest: f32,
}

/// Persistent state of the appendable master file.
#[derive(Debug, Clone)]
pub struct MasterState {
    pub path: String,
    /// End of the last complete fragment (the random-access index follows).
    pub data_end: u64,
    pub total_duration: u64,
    pub next_seq: u32,
    pub timescale: u32,
    pub track_id: u32,
    pub stsd: Vec<u8>,
    pub moov_offset: u64,
    pub moov_size: u64,
    pub movie_timescale: u32,
    /// Segment ids in the order they were appended.
    pub segments: Vec<i64>,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS recordings(
    id INTEGER PRIMARY KEY,
    cam_start INTEGER NOT NULL,
    cam_end INTEGER NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    video_type INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'new',
    file TEXT,
    size INTEGER,
    duration REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    discovered_at INTEGER NOT NULL,
    downloaded_at INTEGER,
    UNIQUE(cam_start, cam_end)
);
CREATE INDEX IF NOT EXISTS recordings_time ON recordings(start_ts, end_ts);
CREATE TABLE IF NOT EXISTS nights(
    night TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    segment_id INTEGER,
    analyzed INTEGER NOT NULL DEFAULT 0,
    emitted INTEGER NOT NULL DEFAULT 0,
    skipped_private INTEGER NOT NULL DEFAULT 0,
    skipped_idle INTEGER NOT NULL DEFAULT 0,
    skipped_dark INTEGER NOT NULL DEFAULT 0,
    nap_frames INTEGER NOT NULL DEFAULT 0,
    skipped_short INTEGER NOT NULL DEFAULT 0,
    best_ts REAL,
    best_score REAL,
    processed_at INTEGER NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS segments(
    id INTEGER PRIMARY KEY,
    night TEXT NOT NULL,
    file TEXT NOT NULL,
    frames INTEGER NOT NULL,
    duration INTEGER NOT NULL,
    appended INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS frames(
    segment_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    ts REAL NOT NULL,
    motion REAL NOT NULL,
    private REAL NOT NULL,
    presence REAL NOT NULL,
    interest REAL NOT NULL,
    PRIMARY KEY(segment_id, idx)
);
CREATE TABLE IF NOT EXISTS masters(
    variant TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    data_end INTEGER NOT NULL,
    total_duration INTEGER NOT NULL,
    next_seq INTEGER NOT NULL,
    timescale INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    stsd BLOB NOT NULL,
    moov_offset INTEGER NOT NULL,
    moov_size INTEGER NOT NULL,
    movie_timescale INTEGER NOT NULL,
    segments TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS master_index(
    variant TEXT NOT NULL,
    seq INTEGER NOT NULL,
    time INTEGER NOT NULL,
    offset INTEGER NOT NULL,
    PRIMARY KEY(variant, seq)
);
CREATE TABLE IF NOT EXISTS redactions(
    id INTEGER PRIMARY KEY,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
"#;

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).with_context(|| format!("cannot open index {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        conn.execute_batch(SCHEMA)?;
        let db = Self { conn };
        // columns added after the first release
        db.ensure_column("nights", "nap_frames", "INTEGER NOT NULL DEFAULT 0")?;
        db.ensure_column("nights", "skipped_short", "INTEGER NOT NULL DEFAULT 0")?;
        db.ensure_column("nights", "uncensored_segment_id", "INTEGER")?;
        db.ensure_column("segments", "variant", "TEXT NOT NULL DEFAULT 'censored'")?;
        db.ensure_column("redactions", "everywhere", "INTEGER NOT NULL DEFAULT 0")?;
        db.migrate_single_master()?;
        Ok(db)
    }

    /// The first release kept one master in `master`/`master_fragments`.
    fn migrate_single_master(&self) -> Result<()> {
        let old: i64 = self.conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='master'", [], |r| r.get(0))?;
        if old == 0 {
            return Ok(());
        }
        let tx = self.conn.unchecked_transaction()?;
        tx.execute_batch(
            "INSERT OR REPLACE INTO masters(variant, path, data_end, total_duration, next_seq, timescale, track_id, stsd,
                                            moov_offset, moov_size, movie_timescale, segments)
               SELECT 'censored', path, data_end, total_duration, next_seq, timescale, track_id, stsd,
                      moov_offset, moov_size, movie_timescale, segments FROM master;
             INSERT OR REPLACE INTO master_index(variant, seq, time, offset)
               SELECT 'censored', seq, time, offset FROM master_fragments;
             DROP TABLE master;
             DROP TABLE master_fragments;",
        )?;
        tx.commit()?;
        log::info!("index upgraded (timelapse variants)");
        Ok(())
    }

    fn ensure_column(&self, table: &str, column: &str, decl: &str) -> Result<()> {
        let mut st = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let exists = st.query_map([], |r| r.get::<_, String>(1))?.filter_map(|c| c.ok()).any(|c| c == column);
        if !exists {
            self.conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {decl}"))?;
        }
        Ok(())
    }

    pub fn meta_get(&self, key: &str) -> Result<Option<String>> {
        Ok(self.conn.query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r.get(0)).optional()?)
    }

    pub fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------- recordings

    /// Inserts a recording if unknown. Returns true when it is new.
    pub fn upsert_recording(&self, cam_start: i64, cam_end: i64, start_ts: i64, end_ts: i64, video_type: i64, now: i64) -> Result<bool> {
        let n = self.conn.execute(
            "INSERT OR IGNORE INTO recordings(cam_start, cam_end, start_ts, end_ts, video_type, discovered_at)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            params![cam_start, cam_end, start_ts, end_ts, video_type, now],
        )?;
        Ok(n > 0)
    }

    fn map_recording(r: &rusqlite::Row) -> rusqlite::Result<Recording> {
        Ok(Recording {
            id: r.get(0)?,
            cam_start: r.get(1)?,
            cam_end: r.get(2)?,
            start_ts: r.get(3)?,
            end_ts: r.get(4)?,
            video_type: r.get(5)?,
            status: RecStatus::parse(&r.get::<_, String>(6)?),
            file: r.get(7)?,
            duration: r.get(8)?,
            attempts: r.get(9)?,
            last_error: r.get(10)?,
        })
    }

    const REC_COLS: &'static str = "id, cam_start, cam_end, start_ts, end_ts, video_type, status, file, duration, attempts, last_error";

    pub fn recordings_with_status(&self, status: RecStatus) -> Result<Vec<Recording>> {
        let sql = format!("SELECT {} FROM recordings WHERE status=?1 ORDER BY start_ts", Self::REC_COLS);
        let mut st = self.conn.prepare(&sql)?;
        let rows = st.query_map([status.as_str()], Self::map_recording)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn recordings_overlapping(&self, start: i64, end: i64) -> Result<Vec<Recording>> {
        let sql = format!("SELECT {} FROM recordings WHERE end_ts > ?1 AND start_ts < ?2 ORDER BY start_ts, end_ts", Self::REC_COLS);
        let mut st = self.conn.prepare(&sql)?;
        let rows = st.query_map(params![start, end], Self::map_recording)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn all_recordings(&self) -> Result<Vec<Recording>> {
        let sql = format!("SELECT {} FROM recordings ORDER BY start_ts", Self::REC_COLS);
        let mut st = self.conn.prepare(&sql)?;
        let rows = st.query_map([], Self::map_recording)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn mark_downloaded(&self, id: i64, file: &str, size: u64, duration: f64, now: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE recordings SET status='downloaded', file=?2, size=?3, duration=?4, downloaded_at=?5, last_error=NULL WHERE id=?1",
            params![id, file, size as i64, duration, now],
        )?;
        Ok(())
    }

    pub fn mark_attempt_failed(&self, id: i64, error: &str, max_attempts: u32) -> Result<RecStatus> {
        self.conn.execute("UPDATE recordings SET attempts=attempts+1, last_error=?2 WHERE id=?1", params![id, error])?;
        let attempts: i64 = self.conn.query_row("SELECT attempts FROM recordings WHERE id=?1", [id], |r| r.get(0))?;
        let status = if attempts >= max_attempts as i64 { RecStatus::Failed } else { RecStatus::New };
        self.set_recording_status(id, status)?;
        Ok(status)
    }

    pub fn set_recording_status(&self, id: i64, status: RecStatus) -> Result<()> {
        self.conn.execute("UPDATE recordings SET status=?2 WHERE id=?1", params![id, status.as_str()])?;
        Ok(())
    }

    pub fn mark_purged(&self, id: i64) -> Result<()> {
        self.conn.execute("UPDATE recordings SET status='purged', file=NULL WHERE id=?1", [id])?;
        Ok(())
    }

    // ---------------------------------------------------------------- nights

    pub fn night_status(&self, night: &str) -> Result<Option<String>> {
        Ok(self.conn.query_row("SELECT status FROM nights WHERE night=?1", [night], |r| r.get(0)).optional()?)
    }

    pub fn first_night(&self) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT MIN(night) FROM nights WHERE status IN ('processed', 'reprocess')", [], |r| r.get(0))
            .optional()?
            .flatten())
    }

    /// (night, status, statistics, segment id)
    pub fn list_nights(&self) -> Result<Vec<NightRow>> {
        let mut st = self.conn.prepare(
            "SELECT n.night, n.status, n.analyzed, n.emitted, n.skipped_private, n.skipped_idle, n.skipped_dark, n.best_ts,
                    n.best_score, n.segment_id, n.nap_frames, n.skipped_short, n.uncensored_segment_id, COALESCE(u.frames, 0)
             FROM nights n LEFT JOIN segments u ON u.id = n.uncensored_segment_id ORDER BY n.night",
        )?;
        let rows = st.query_map([], |r| {
            Ok(NightRow {
                night: r.get(0)?,
                status: r.get(1)?,
                stats: NightStats {
                    analyzed: r.get(2)?,
                    emitted: r.get(3)?,
                    skipped_private: r.get(4)?,
                    skipped_idle: r.get(5)?,
                    skipped_dark: r.get(6)?,
                    nap_frames: r.get(10)?,
                    skipped_short: r.get(11)?,
                    best_ts: r.get(7)?,
                    best_score: r.get(8)?,
                },
                segment: r.get(9)?,
                uncensored_segment: r.get(12)?,
                uncensored_frames: r.get(13)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_night(
        &self,
        night: &str,
        status: &str,
        segment_id: Option<i64>,
        uncensored_id: Option<i64>,
        s: &NightStats,
        now: i64,
        error: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO nights(night, status, segment_id, analyzed, emitted, skipped_private, skipped_idle, skipped_dark, best_ts, best_score,
                                processed_at, error, nap_frames, skipped_short, uncensored_segment_id)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
             ON CONFLICT(night) DO UPDATE SET status=excluded.status, segment_id=excluded.segment_id, analyzed=excluded.analyzed,
               emitted=excluded.emitted, skipped_private=excluded.skipped_private, skipped_idle=excluded.skipped_idle,
               skipped_dark=excluded.skipped_dark, best_ts=excluded.best_ts, best_score=excluded.best_score,
               processed_at=excluded.processed_at, error=excluded.error, nap_frames=excluded.nap_frames,
               skipped_short=excluded.skipped_short, uncensored_segment_id=excluded.uncensored_segment_id",
            params![
                night,
                status,
                segment_id,
                s.analyzed,
                s.emitted,
                s.skipped_private,
                s.skipped_idle,
                s.skipped_dark,
                s.best_ts,
                s.best_score,
                now,
                error,
                s.nap_frames,
                s.skipped_short,
                uncensored_id
            ],
        )?;
        Ok(())
    }

    /// The uncensored segment of a night, if it has one.
    pub fn uncensored_segment_of(&self, night: &str) -> Result<Option<i64>> {
        Ok(self.conn.query_row("SELECT uncensored_segment_id FROM nights WHERE night=?1", [night], |r| r.get(0)).optional()?.flatten())
    }

    /// Changes the status of a known night, keeping its segments. Returns false for an unknown night.
    pub fn set_night_status(&self, night: &str, status: &str, error: Option<&str>) -> Result<bool> {
        let n = self.conn.execute("UPDATE nights SET status=?2, error=?3 WHERE night=?1", params![night, status, error])?;
        Ok(n > 0)
    }

    // ---------------------------------------------------------------- segments

    pub fn insert_segment(&self, night: &str, variant: Variant, file: &str, frames: &[FrameRow], duration: i64, now: i64) -> Result<i64> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO segments(night, file, frames, duration, appended, created_at, variant) VALUES(?1, ?2, ?3, ?4, 0, ?5, ?6)",
            params![night, file, frames.len() as i64, duration, now, variant.key()],
        )?;
        let id = tx.last_insert_rowid();
        {
            let mut st = tx.prepare(
                "INSERT INTO frames(segment_id, idx, ts, motion, private, presence, interest) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for f in frames {
                st.execute(params![id, f.idx, f.ts, f.motion, f.private, f.presence, f.interest])?;
            }
        }
        tx.commit()?;
        Ok(id)
    }

    /// Replaces the content of a segment (after a redaction re-encode).
    pub fn replace_segment_frames(&self, id: i64, file: &str, frames: &[FrameRow], duration: i64) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM frames WHERE segment_id=?1", [id])?;
        {
            let mut st = tx.prepare(
                "INSERT INTO frames(segment_id, idx, ts, motion, private, presence, interest) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for f in frames {
                st.execute(params![id, f.idx, f.ts, f.motion, f.private, f.presence, f.interest])?;
            }
        }
        tx.execute(
            "UPDATE segments SET file=?2, frames=?3, duration=?4, appended=0 WHERE id=?1",
            params![id, file, frames.len() as i64, duration],
        )?;
        tx.commit()?;
        Ok(())
    }

    fn map_segment(r: &rusqlite::Row) -> rusqlite::Result<Segment> {
        Ok(Segment {
            id: r.get(0)?,
            night: r.get(1)?,
            file: r.get(2)?,
            frames: r.get(3)?,
            duration: r.get(4)?,
            appended: r.get::<_, i64>(5)? != 0,
        })
    }

    /// Segments that make up a timelapse, in chronological order (one per night).
    pub fn active_segments(&self, variant: Variant) -> Result<Vec<Segment>> {
        let mut st = self.conn.prepare(&format!(
            "SELECT s.id, s.night, s.file, s.frames, s.duration, s.appended FROM segments s
             JOIN nights n ON n.{} = s.id AND n.status IN ('processed', 'reprocess')
             WHERE s.frames > 0 ORDER BY s.night, s.id",
            variant.night_column()
        ))?;
        let rows = st.query_map([], Self::map_segment)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn segment_frames(&self, id: i64) -> Result<Vec<FrameRow>> {
        let mut st =
            self.conn.prepare("SELECT idx, ts, motion, private, presence, interest FROM frames WHERE segment_id=?1 ORDER BY idx")?;
        let rows = st.query_map([id], |r| {
            Ok(FrameRow {
                idx: r.get(0)?,
                ts: r.get(1)?,
                motion: r.get::<_, f64>(2)? as f32,
                private: r.get::<_, f64>(3)? as f32,
                presence: r.get::<_, f64>(4)? as f32,
                interest: r.get::<_, f64>(5)? as f32,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn segments_unreferenced(&self) -> Result<Vec<Segment>> {
        let mut st = self.conn.prepare(
            "SELECT id, night, file, frames, duration, appended FROM segments
             WHERE id NOT IN (SELECT segment_id FROM nights WHERE segment_id IS NOT NULL
                              UNION SELECT uncensored_segment_id FROM nights WHERE uncensored_segment_id IS NOT NULL)",
        )?;
        let rows = st.query_map([], Self::map_segment)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn delete_segment(&self, id: i64) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM frames WHERE segment_id=?1", [id])?;
        tx.execute("DELETE FROM segments WHERE id=?1", [id])?;
        tx.commit()?;
        Ok(())
    }

    // ---------------------------------------------------------------- master

    pub fn master(&self, variant: Variant) -> Result<Option<MasterState>> {
        Ok(self
            .conn
            .query_row(
                "SELECT path, data_end, total_duration, next_seq, timescale, track_id, stsd, moov_offset, moov_size, movie_timescale, segments
                 FROM masters WHERE variant=?1",
                [variant.key()],
                |r| {
                    Ok(MasterState {
                        path: r.get(0)?,
                        data_end: r.get::<_, i64>(1)? as u64,
                        total_duration: r.get::<_, i64>(2)? as u64,
                        next_seq: r.get::<_, i64>(3)? as u32,
                        timescale: r.get::<_, i64>(4)? as u32,
                        track_id: r.get::<_, i64>(5)? as u32,
                        stsd: r.get(6)?,
                        moov_offset: r.get::<_, i64>(7)? as u64,
                        moov_size: r.get::<_, i64>(8)? as u64,
                        movie_timescale: r.get::<_, i64>(9)? as u32,
                        segments: r
                            .get::<_, String>(10)?
                            .split(',')
                            .filter_map(|x| x.trim().parse().ok())
                            .collect(),
                    })
                },
            )
            .optional()?)
    }

    pub fn master_fragments(&self, variant: Variant) -> Result<Vec<(u64, u64)>> {
        let mut st = self.conn.prepare("SELECT time, offset FROM master_index WHERE variant=?1 ORDER BY seq")?;
        let rows = st.query_map([variant.key()], |r| Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Atomically records a successful append (or a fresh rebuild when `reset`).
    pub fn commit_master(&self, variant: Variant, m: &MasterState, new_fragments: &[(u32, u64, u64)], reset: bool) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        if reset {
            tx.execute("DELETE FROM master_index WHERE variant=?1", [variant.key()])?;
            tx.execute("UPDATE segments SET appended=0 WHERE variant=?1", [variant.key()])?;
        }
        tx.execute(
            "INSERT INTO masters(variant, path, data_end, total_duration, next_seq, timescale, track_id, stsd, moov_offset, moov_size, movie_timescale, segments)
             VALUES(?12, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(variant) DO UPDATE SET path=excluded.path, data_end=excluded.data_end, total_duration=excluded.total_duration,
               next_seq=excluded.next_seq, timescale=excluded.timescale, track_id=excluded.track_id, stsd=excluded.stsd,
               moov_offset=excluded.moov_offset, moov_size=excluded.moov_size, movie_timescale=excluded.movie_timescale,
               segments=excluded.segments",
            params![
                m.path,
                m.data_end as i64,
                m.total_duration as i64,
                m.next_seq as i64,
                m.timescale as i64,
                m.track_id as i64,
                m.stsd,
                m.moov_offset as i64,
                m.moov_size as i64,
                m.movie_timescale as i64,
                m.segments.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
                variant.key()
            ],
        )?;
        {
            let mut st = tx.prepare("INSERT OR REPLACE INTO master_index(variant, seq, time, offset) VALUES(?1, ?2, ?3, ?4)")?;
            for (seq, time, off) in new_fragments {
                st.execute(params![variant.key(), *seq as i64, *time as i64, *off as i64])?;
            }
        }
        for id in &m.segments {
            tx.execute("UPDATE segments SET appended=1 WHERE id=?1", [id])?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn clear_master(&self, variant: Variant) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM masters WHERE variant=?1", [variant.key()])?;
        tx.execute("DELETE FROM master_index WHERE variant=?1", [variant.key()])?;
        tx.execute("UPDATE segments SET appended=0 WHERE variant=?1", [variant.key()])?;
        tx.commit()?;
        Ok(())
    }

    // ---------------------------------------------------------------- redactions

    /// `everywhere`: also cut from the uncensored timelapse (by default a
    /// redaction is a correction of the privacy filter and only affects the regular one).
    pub fn add_redaction(&self, start: i64, end: i64, everywhere: bool, now: i64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO redactions(start_ts, end_ts, created_at, everywhere) VALUES(?1, ?2, ?3, ?4)",
            params![start, end, now, everywhere as i64],
        )?;
        Ok(())
    }

    pub fn redactions(&self, variant: Variant) -> Result<Vec<(i64, i64)>> {
        let only_everywhere = variant == Variant::Uncensored;
        let mut st = self.conn.prepare("SELECT start_ts, end_ts FROM redactions WHERE everywhere=1 OR ?1=0 ORDER BY start_ts")?;
        let rows = st.query_map([only_everywhere as i64], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(idx: i64) -> FrameRow {
        FrameRow { idx, ts: idx as f64, motion: 0.0, private: 0.0, presence: 0.0, interest: 0.0 }
    }

    #[test]
    fn single_master_index_is_migrated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        {
            // the layout of the first release
            let c = Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE master(id INTEGER PRIMARY KEY CHECK (id = 1), path TEXT NOT NULL, data_end INTEGER NOT NULL,
                   total_duration INTEGER NOT NULL, next_seq INTEGER NOT NULL, timescale INTEGER NOT NULL, track_id INTEGER NOT NULL,
                   stsd BLOB NOT NULL, moov_offset INTEGER NOT NULL, moov_size INTEGER NOT NULL, movie_timescale INTEGER NOT NULL,
                   segments TEXT NOT NULL);
                 CREATE TABLE master_fragments(seq INTEGER PRIMARY KEY, time INTEGER NOT NULL, offset INTEGER NOT NULL);
                 CREATE TABLE segments(id INTEGER PRIMARY KEY, night TEXT NOT NULL, file TEXT NOT NULL, frames INTEGER NOT NULL,
                   duration INTEGER NOT NULL, appended INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
                 CREATE TABLE redactions(id INTEGER PRIMARY KEY, start_ts INTEGER NOT NULL, end_ts INTEGER NOT NULL,
                   created_at INTEGER NOT NULL);
                 INSERT INTO master VALUES(1, '/x/timelapse.mp4', 1000, 90000, 3, 90000, 1, x'00', 32, 500, 1000, '6,2');
                 INSERT INTO master_fragments VALUES(1, 0, 600), (2, 45000, 800);
                 INSERT INTO segments VALUES(6, '2026-10-10', 'a.mp4', 10, 3000, 1, 0), (2, '2026-10-11', 'b.mp4', 10, 3000, 1, 0);
                 INSERT INTO redactions VALUES(1, 100, 200, 0);",
            )
            .unwrap();
        }
        let db = Db::open(&path).unwrap();
        let m = db.master(Variant::Censored).unwrap().unwrap();
        assert_eq!((m.path.as_str(), m.data_end, m.segments.clone()), ("/x/timelapse.mp4", 1000, vec![6, 2]));
        assert_eq!(db.master_fragments(Variant::Censored).unwrap(), vec![(0, 600), (45000, 800)]);
        assert!(db.master(Variant::Uncensored).unwrap().is_none());
        // old redactions were corrections of the privacy filter
        assert_eq!(db.redactions(Variant::Censored).unwrap(), vec![(100, 200)]);
        assert!(db.redactions(Variant::Uncensored).unwrap().is_empty());
        // opening again is a no-op
        drop(db);
        let db = Db::open(&path).unwrap();
        assert_eq!(db.master(Variant::Censored).unwrap().unwrap().segments, vec![6, 2]);
    }

    #[test]
    fn variants_keep_separate_segments_and_masters() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("index.db")).unwrap();
        let frames: Vec<_> = (0..3).map(frame).collect();
        let a = db.insert_segment("2026-10-10", Variant::Censored, "a.mp4", &frames[..1], 3000, 0).unwrap();
        let b = db.insert_segment("2026-10-10", Variant::Uncensored, "b.mp4", &frames, 9000, 0).unwrap();
        db.set_night("2026-10-10", "processed", Some(a), Some(b), &NightStats::default(), 0, None).unwrap();
        assert_eq!(db.active_segments(Variant::Censored).unwrap().iter().map(|s| s.id).collect::<Vec<_>>(), vec![a]);
        assert_eq!(db.active_segments(Variant::Uncensored).unwrap().iter().map(|s| s.id).collect::<Vec<_>>(), vec![b]);
        assert!(db.segments_unreferenced().unwrap().is_empty());
        let row = &db.list_nights().unwrap()[0];
        assert_eq!((row.segment, row.uncensored_segment, row.uncensored_frames), (Some(a), Some(b), 3));

        let state = |segments| MasterState {
            path: "m.mp4".into(),
            data_end: 1,
            total_duration: 2,
            next_seq: 3,
            timescale: 90000,
            track_id: 1,
            stsd: vec![0],
            moov_offset: 0,
            moov_size: 0,
            movie_timescale: 1000,
            segments,
        };
        db.commit_master(Variant::Censored, &state(vec![a]), &[(1, 0, 10)], false).unwrap();
        db.commit_master(Variant::Uncensored, &state(vec![b]), &[(1, 0, 20), (2, 5, 30)], false).unwrap();
        // a rebuild of one version leaves the other alone
        db.commit_master(Variant::Censored, &state(vec![a]), &[(1, 0, 11)], true).unwrap();
        assert_eq!(db.master_fragments(Variant::Censored).unwrap(), vec![(0, 11)]);
        assert_eq!(db.master_fragments(Variant::Uncensored).unwrap(), vec![(0, 20), (5, 30)]);
        db.clear_master(Variant::Censored).unwrap();
        assert!(db.master(Variant::Censored).unwrap().is_none());
        assert_eq!(db.master(Variant::Uncensored).unwrap().unwrap().segments, vec![b]);

        // a night being reprocessed keeps its segments until it is replaced
        assert!(db.set_night_status("2026-10-10", "reprocess", None).unwrap());
        assert_eq!(db.active_segments(Variant::Uncensored).unwrap().len(), 1);
        assert!(!db.set_night_status("2026-01-01", "reprocess", None).unwrap());

        db.add_redaction(10, 20, false, 0).unwrap();
        db.add_redaction(30, 40, true, 0).unwrap();
        assert_eq!(db.redactions(Variant::Censored).unwrap(), vec![(10, 20), (30, 40)]);
        assert_eq!(db.redactions(Variant::Uncensored).unwrap(), vec![(30, 40)]);
    }
}
