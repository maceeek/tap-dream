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

pub type NightRow = (String, String, NightStats, Option<i64>);

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
CREATE TABLE IF NOT EXISTS master(
    id INTEGER PRIMARY KEY CHECK (id = 1),
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
CREATE TABLE IF NOT EXISTS master_fragments(
    seq INTEGER PRIMARY KEY,
    time INTEGER NOT NULL,
    offset INTEGER NOT NULL
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
        Ok(db)
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
        Ok(self.conn.query_row("SELECT MIN(night) FROM nights WHERE status='processed'", [], |r| r.get(0)).optional()?.flatten())
    }

    /// (night, status, statistics, segment id)
    pub fn list_nights(&self) -> Result<Vec<NightRow>> {
        let mut st = self.conn.prepare(
            "SELECT night, status, analyzed, emitted, skipped_private, skipped_idle, skipped_dark, best_ts, best_score, segment_id,
                    nap_frames, skipped_short
             FROM nights ORDER BY night",
        )?;
        let rows = st.query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                NightStats {
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
                r.get(9)?,
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn set_night(
        &self,
        night: &str,
        status: &str,
        segment_id: Option<i64>,
        s: &NightStats,
        now: i64,
        error: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO nights(night, status, segment_id, analyzed, emitted, skipped_private, skipped_idle, skipped_dark, best_ts, best_score,
                                processed_at, error, nap_frames, skipped_short)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
             ON CONFLICT(night) DO UPDATE SET status=excluded.status, segment_id=excluded.segment_id, analyzed=excluded.analyzed,
               emitted=excluded.emitted, skipped_private=excluded.skipped_private, skipped_idle=excluded.skipped_idle,
               skipped_dark=excluded.skipped_dark, best_ts=excluded.best_ts, best_score=excluded.best_score,
               processed_at=excluded.processed_at, error=excluded.error, nap_frames=excluded.nap_frames,
               skipped_short=excluded.skipped_short",
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
                s.skipped_short
            ],
        )?;
        Ok(())
    }

    pub fn delete_night(&self, night: &str) -> Result<()> {
        self.conn.execute("DELETE FROM nights WHERE night=?1", [night])?;
        Ok(())
    }

    // ---------------------------------------------------------------- segments

    pub fn insert_segment(&self, night: &str, file: &str, frames: &[FrameRow], duration: i64, now: i64) -> Result<i64> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO segments(night, file, frames, duration, appended, created_at) VALUES(?1, ?2, ?3, ?4, 0, ?5)",
            params![night, file, frames.len() as i64, duration, now],
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

    /// Segments that make up the timelapse, in chronological order (one per night).
    pub fn active_segments(&self) -> Result<Vec<Segment>> {
        let mut st = self.conn.prepare(
            "SELECT s.id, s.night, s.file, s.frames, s.duration, s.appended FROM segments s
             JOIN nights n ON n.segment_id = s.id AND n.status='processed'
             WHERE s.frames > 0 ORDER BY s.night, s.id",
        )?;
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
             WHERE id NOT IN (SELECT segment_id FROM nights WHERE segment_id IS NOT NULL)",
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

    pub fn master(&self) -> Result<Option<MasterState>> {
        Ok(self
            .conn
            .query_row(
                "SELECT path, data_end, total_duration, next_seq, timescale, track_id, stsd, moov_offset, moov_size, movie_timescale, segments FROM master WHERE id=1",
                [],
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

    pub fn master_fragments(&self) -> Result<Vec<(u64, u64)>> {
        let mut st = self.conn.prepare("SELECT time, offset FROM master_fragments ORDER BY seq")?;
        let rows = st.query_map([], |r| Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Atomically records a successful append (or a fresh rebuild when `reset`).
    pub fn commit_master(&self, m: &MasterState, new_fragments: &[(u32, u64, u64)], reset: bool) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        if reset {
            tx.execute("DELETE FROM master_fragments", [])?;
            tx.execute("UPDATE segments SET appended=0", [])?;
        }
        tx.execute(
            "INSERT INTO master(id, path, data_end, total_duration, next_seq, timescale, track_id, stsd, moov_offset, moov_size, movie_timescale, segments)
             VALUES(1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(id) DO UPDATE SET path=excluded.path, data_end=excluded.data_end, total_duration=excluded.total_duration,
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
                m.segments.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
            ],
        )?;
        {
            let mut st = tx.prepare("INSERT OR REPLACE INTO master_fragments(seq, time, offset) VALUES(?1, ?2, ?3)")?;
            for (seq, time, off) in new_fragments {
                st.execute(params![*seq as i64, *time as i64, *off as i64])?;
            }
        }
        for id in &m.segments {
            tx.execute("UPDATE segments SET appended=1 WHERE id=?1", [id])?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn clear_master(&self) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM master", [])?;
        tx.execute("DELETE FROM master_fragments", [])?;
        tx.execute("UPDATE segments SET appended=0", [])?;
        tx.commit()?;
        Ok(())
    }

    // ---------------------------------------------------------------- redactions

    pub fn add_redaction(&self, start: i64, end: i64, now: i64) -> Result<()> {
        self.conn.execute("INSERT INTO redactions(start_ts, end_ts, created_at) VALUES(?1, ?2, ?3)", params![start, end, now])?;
        Ok(())
    }

    pub fn redactions(&self) -> Result<Vec<(i64, i64)>> {
        let mut st = self.conn.prepare("SELECT start_ts, end_ts FROM redactions ORDER BY start_ts")?;
        let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}
