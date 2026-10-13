//! Incremental MPEG-TS inspection: measures how much video (by PTS) has been
//! received, so a download can be validated without spawning ffprobe.

const PACKET: usize = 188;

#[derive(Default)]
pub struct TsDuration {
    carry: Vec<u8>,
    pmt_pid: Option<u16>,
    video_pid: Option<u16>,
    first_pts: Option<u64>,
    last_pts: u64,
    /// Accumulated PTS span (handles the 33-bit wrap).
    span: u64,
    prev_pts: Option<u64>,
    pub packets: u64,
}

impl TsDuration {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seconds of video seen so far.
    pub fn seconds(&self) -> f64 {
        self.span as f64 / 90_000.0
    }

    pub fn feed(&mut self, data: &[u8]) {
        let mut buf = std::mem::take(&mut self.carry);
        buf.extend_from_slice(data);
        let mut i = 0;
        while i + PACKET <= buf.len() {
            if buf[i] != 0x47 {
                i += 1; // resync
                continue;
            }
            self.packet(&buf[i..i + PACKET]);
            i += PACKET;
        }
        self.carry = buf[i..].to_vec();
    }

    fn packet(&mut self, p: &[u8]) {
        self.packets += 1;
        let pusi = p[1] & 0x40 != 0;
        let pid = (((p[1] & 0x1f) as u16) << 8) | p[2] as u16;
        let afc = (p[3] >> 4) & 0x3;
        let mut off = 4;
        if afc & 0x2 != 0 {
            off += 1 + p[4] as usize;
        }
        if afc & 0x1 == 0 || off >= PACKET {
            return;
        }
        let payload = &p[off..];
        if pid == 0 && pusi {
            // PAT
            let ptr = payload[0] as usize;
            let t = &payload[(1 + ptr).min(payload.len())..];
            if t.len() < 8 {
                return;
            }
            let section_len = (((t[1] & 0x0f) as usize) << 8) | t[2] as usize;
            let end = (3 + section_len).min(t.len()).saturating_sub(4);
            let mut j = 8;
            while j + 4 <= end {
                let program = ((t[j] as u16) << 8) | t[j + 1] as u16;
                let ppid = (((t[j + 2] & 0x1f) as u16) << 8) | t[j + 3] as u16;
                if program != 0 {
                    self.pmt_pid = Some(ppid);
                }
                j += 4;
            }
        } else if Some(pid) == self.pmt_pid && pusi && self.video_pid.is_none() {
            let ptr = payload[0] as usize;
            let t = &payload[(1 + ptr).min(payload.len())..];
            if t.len() < 12 {
                return;
            }
            let section_len = (((t[1] & 0x0f) as usize) << 8) | t[2] as usize;
            let end = (3 + section_len).min(t.len()).saturating_sub(4);
            let info_len = (((t[10] & 0x0f) as usize) << 8) | t[11] as usize;
            let mut j = 12 + info_len;
            while j + 5 <= end {
                let stype = t[j];
                let epid = (((t[j + 1] & 0x1f) as u16) << 8) | t[j + 2] as u16;
                let es_len = (((t[j + 3] & 0x0f) as usize) << 8) | t[j + 4] as usize;
                if matches!(stype, 0x1b | 0x24) {
                    self.video_pid = Some(epid);
                    break;
                }
                j += 5 + es_len;
            }
        } else if Some(pid) == self.video_pid && pusi {
            if payload.len() < 14 || payload[0] != 0 || payload[1] != 0 || payload[2] != 1 {
                return;
            }
            if payload[7] & 0x80 == 0 {
                return; // no PTS
            }
            let b = &payload[9..14];
            let pts = (((b[0] as u64 >> 1) & 0x07) << 30)
                | ((b[1] as u64) << 22)
                | (((b[2] as u64) >> 1) << 15)
                | ((b[3] as u64) << 7)
                | ((b[4] as u64) >> 1);
            self.add_pts(pts);
        }
    }

    fn add_pts(&mut self, pts: u64) {
        const WRAP: u64 = 1 << 33;
        if self.first_pts.is_none() {
            self.first_pts = Some(pts);
        }
        if let Some(prev) = self.prev_pts {
            let delta = (pts + WRAP - prev) % WRAP;
            // Only count forward steps of plausible size (B-frame reordering
            // produces small backward steps that we ignore).
            if delta < 90_000 * 10 {
                self.last_pts = pts;
                self.span += delta;
                self.prev_pts = Some(pts);
            } else if (prev + WRAP - pts) % WRAP < 90_000 * 10 {
                // backward step: ignore
            } else {
                // discontinuity: restart the measurement from here
                self.prev_pts = Some(pts);
            }
        } else {
            self.prev_pts = Some(pts);
            self.last_pts = pts;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_generated_ts() {
        // 4 s of 10 fps H.264 produced by ffmpeg (skips when ffmpeg is missing).
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("t.ts");
        let ok = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=4", "-c:v", "libx264", "-f", "mpegts"])
            .arg(&out)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            return;
        }
        let data = std::fs::read(&out).unwrap();
        let mut d = TsDuration::new();
        for chunk in data.chunks(1000) {
            d.feed(chunk);
        }
        assert!((d.seconds() - 3.9).abs() < 0.15, "{}", d.seconds());
    }
}
