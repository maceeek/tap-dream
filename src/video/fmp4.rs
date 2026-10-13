//! Fragmented MP4 surgery: the timelapse is ONE file that grows by appending
//! the fragments of each new night's segment. Only the new data is written;
//! the small header (moov) is patched in place with the new duration and a
//! random-access index (mfra) is kept at the end so players can seek.
//!
//! File layout:  ftyp | moov (64-bit durations) | (moof mdat)* | mfra
//! `data_end` (stored in the index DB) points just past the last mdat, so an
//! interrupted append is undone by truncating back to it.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const CONTAINERS: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"mvex", b"edts", b"dinf", b"moof", b"traf", b"mfra"];

#[derive(Clone, Debug)]
pub enum Mp4Box {
    Leaf { typ: [u8; 4], payload: Vec<u8> },
    Container { typ: [u8; 4], children: Vec<Mp4Box> },
}

impl Mp4Box {
    pub fn typ(&self) -> &[u8; 4] {
        match self {
            Mp4Box::Leaf { typ, .. } | Mp4Box::Container { typ, .. } => typ,
        }
    }

    pub fn encoded_len(&self) -> usize {
        8 + match self {
            Mp4Box::Leaf { payload, .. } => payload.len(),
            Mp4Box::Container { children, .. } => children.iter().map(Mp4Box::encoded_len).sum(),
        }
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.encoded_len() as u32).to_be_bytes());
        out.extend_from_slice(self.typ());
        match self {
            Mp4Box::Leaf { payload, .. } => out.extend_from_slice(payload),
            Mp4Box::Container { children, .. } => children.iter().for_each(|c| c.write_to(out)),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.encoded_len());
        self.write_to(&mut v);
        v
    }

    fn children_mut(&mut self) -> Option<&mut Vec<Mp4Box>> {
        match self {
            Mp4Box::Container { children, .. } => Some(children),
            _ => None,
        }
    }

    /// Depth-first search by path, e.g. ["trak", "mdia", "mdhd"].
    pub fn find(&self, path: &[&[u8; 4]]) -> Option<&Mp4Box> {
        if path.is_empty() {
            return Some(self);
        }
        if let Mp4Box::Container { children, .. } = self {
            for c in children {
                if c.typ() == path[0] {
                    if let Some(f) = c.find(&path[1..]) {
                        return Some(f);
                    }
                }
            }
        }
        None
    }

    pub fn find_mut(&mut self, path: &[&[u8; 4]]) -> Option<&mut Mp4Box> {
        if path.is_empty() {
            return Some(self);
        }
        if let Mp4Box::Container { children, .. } = self {
            for c in children.iter_mut() {
                if c.typ() == path[0] {
                    return c.find_mut(&path[1..]);
                }
            }
        }
        None
    }

    pub fn payload(&self) -> Option<&[u8]> {
        match self {
            Mp4Box::Leaf { payload, .. } => Some(payload),
            _ => None,
        }
    }

    pub fn payload_mut(&mut self) -> Option<&mut Vec<u8>> {
        match self {
            Mp4Box::Leaf { payload, .. } => Some(payload),
            _ => None,
        }
    }
}

/// (type, start offset, header length, total size)
pub type BoxPos = ([u8; 4], usize, usize, usize);

/// (sequence number, decode time, file offset) of a fragment written to the master
pub type FragmentEntry = (u32, u64, u64);

/// Top-level box positions.
pub fn top_level(data: &[u8]) -> Result<Vec<BoxPos>> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 8 <= data.len() {
        let (size, hl) = box_size(&data[off..], data.len() - off)?;
        let typ: [u8; 4] = data[off + 4..off + 8].try_into().unwrap();
        if off + size > data.len() {
            bail!("truncated {} box", String::from_utf8_lossy(&typ));
        }
        out.push((typ, off, hl, size));
        off += size;
    }
    Ok(out)
}

fn box_size(d: &[u8], remaining: usize) -> Result<(usize, usize)> {
    let size = u32::from_be_bytes(d[..4].try_into().unwrap()) as u64;
    let (size, hl) = match size {
        1 => {
            if d.len() < 16 {
                bail!("truncated large box header");
            }
            (u64::from_be_bytes(d[8..16].try_into().unwrap()), 16)
        }
        0 => (remaining as u64, 8),
        s => (s, 8),
    };
    if size < hl as u64 {
        bail!("invalid box size {size}");
    }
    Ok((size as usize, hl))
}

pub fn parse_box(data: &[u8]) -> Result<Mp4Box> {
    let (size, hl) = box_size(data, data.len())?;
    let typ: [u8; 4] = data[4..8].try_into().unwrap();
    let body = &data[hl..size.min(data.len())];
    if CONTAINERS.contains(&&typ) {
        let mut children = Vec::new();
        let mut off = 0;
        while off + 8 <= body.len() {
            let (csize, _) = box_size(&body[off..], body.len() - off)?;
            children.push(parse_box(&body[off..off + csize.min(body.len() - off)])?);
            off += csize;
        }
        Ok(Mp4Box::Container { typ, children })
    } else {
        Ok(Mp4Box::Leaf { typ, payload: body.to_vec() })
    }
}

fn be32(p: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(p[o..o + 4].try_into().unwrap())
}
fn be64(p: &[u8], o: usize) -> u64 {
    u64::from_be_bytes(p[o..o + 8].try_into().unwrap())
}

/// Converts mvhd/tkhd/mdhd (version 0, 32-bit durations) to version 1 so the
/// duration can grow for decades. Layout: version+flags, creation, modification,
/// then box specific fields.
fn to_v1(payload: &[u8], kind: &[u8; 4]) -> Vec<u8> {
    if payload[0] == 1 {
        return payload.to_vec();
    }
    let mut out = vec![1, payload[1], payload[2], payload[3]];
    let creation = be32(payload, 4) as u64;
    let modification = be32(payload, 8) as u64;
    out.extend_from_slice(&creation.to_be_bytes());
    out.extend_from_slice(&modification.to_be_bytes());
    match kind {
        b"tkhd" => {
            out.extend_from_slice(&payload[12..20]); // track_ID, reserved
            out.extend_from_slice(&(be32(payload, 20) as u64).to_be_bytes());
            out.extend_from_slice(&payload[24..]);
        }
        _ => {
            // mvhd / mdhd: timescale, duration
            out.extend_from_slice(&payload[12..16]);
            out.extend_from_slice(&(be32(payload, 16) as u64).to_be_bytes());
            out.extend_from_slice(&payload[20..]);
        }
    }
    out
}

fn set_v1_duration(payload: &mut [u8], kind: &[u8; 4], duration: u64) {
    let off = if kind == b"tkhd" { 28 } else { 24 };
    payload[off..off + 8].copy_from_slice(&duration.to_be_bytes());
}

/// What we need to know about a segment's single video track.
#[derive(Clone, Debug)]
pub struct TrackInfo {
    pub track_id: u32,
    pub timescale: u32,
    pub movie_timescale: u32,
    pub stsd: Vec<u8>,
    pub trex_duration: u32,
    pub trex_size: u32,
    pub trex_flags: u32,
}

pub fn track_info(moov: &Mp4Box) -> Result<TrackInfo> {
    let mvhd = moov.find(&[b"mvhd"]).and_then(Mp4Box::payload).ok_or_else(|| anyhow!("no mvhd"))?;
    let movie_timescale = if mvhd[0] == 1 { be32(mvhd, 20) } else { be32(mvhd, 12) };
    let tkhd = moov.find(&[b"trak", b"tkhd"]).and_then(Mp4Box::payload).ok_or_else(|| anyhow!("no tkhd"))?;
    let track_id = if tkhd[0] == 1 { be32(tkhd, 20) } else { be32(tkhd, 12) };
    let mdhd = moov.find(&[b"trak", b"mdia", b"mdhd"]).and_then(Mp4Box::payload).ok_or_else(|| anyhow!("no mdhd"))?;
    let timescale = if mdhd[0] == 1 { be32(mdhd, 20) } else { be32(mdhd, 12) };
    let stsd =
        moov.find(&[b"trak", b"mdia", b"minf", b"stbl", b"stsd"]).and_then(Mp4Box::payload).ok_or_else(|| anyhow!("no stsd"))?.to_vec();
    let trex = moov.find(&[b"mvex", b"trex"]).and_then(Mp4Box::payload).ok_or_else(|| anyhow!("segment is not fragmented (no trex)"))?;
    if let Mp4Box::Container { children, .. } = moov {
        if children.iter().filter(|c| c.typ() == b"trak").count() != 1 {
            bail!("segment must have exactly one track");
        }
    }
    Ok(TrackInfo {
        track_id,
        timescale,
        movie_timescale,
        stsd,
        trex_duration: be32(trex, 12),
        trex_size: be32(trex, 16),
        trex_flags: be32(trex, 20),
    })
}

/// The init part of the master file, derived from the first segment's moov.
pub fn master_moov(moov: &Mp4Box) -> Result<Mp4Box> {
    let mut m = moov.clone();
    for (path, kind) in [(vec![b"mvhd"], b"mvhd"), (vec![b"trak", b"tkhd"], b"tkhd"), (vec![b"trak", b"mdia", b"mdhd"], b"mdhd")] {
        let b = m.find_mut(&path).and_then(Mp4Box::payload_mut).ok_or_else(|| anyhow!("missing {}", String::from_utf8_lossy(kind)))?;
        *b = to_v1(b, kind);
    }
    // An edit list from a single segment would cut the presentation short.
    if let Some(trak) = m.find_mut(&[b"trak"]).and_then(Mp4Box::children_mut) {
        trak.retain(|c| c.typ() != b"edts");
    }
    let mvex = m.find_mut(&[b"mvex"]).and_then(Mp4Box::children_mut).ok_or_else(|| anyhow!("no mvex"))?;
    mvex.retain(|c| c.typ() != b"mehd");
    mvex.insert(0, Mp4Box::Leaf { typ: *b"mehd", payload: vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
    Ok(m)
}

pub fn set_master_duration(moov: &mut Mp4Box, track_duration: u64, timescale: u32, movie_timescale: u32) -> Result<()> {
    let movie = (track_duration as u128 * movie_timescale as u128 / timescale.max(1) as u128) as u64;
    for (path, kind, value) in [
        (vec![b"mvhd"], b"mvhd", movie),
        (vec![b"trak", b"tkhd"], b"tkhd", movie),
        (vec![b"trak", b"mdia", b"mdhd"], b"mdhd", track_duration),
    ] {
        let p = moov.find_mut(&path).and_then(Mp4Box::payload_mut).ok_or_else(|| anyhow!("missing box"))?;
        set_v1_duration(p, kind, value);
    }
    let mehd = moov.find_mut(&[b"mvex", b"mehd"]).and_then(Mp4Box::payload_mut).ok_or_else(|| anyhow!("missing mehd"))?;
    mehd[4..12].copy_from_slice(&movie.to_be_bytes());
    Ok(())
}

/// A parsed segment file kept in memory (segments are small).
pub struct Segment {
    pub ftyp: Vec<u8>,
    pub moov: Mp4Box,
    pub info: TrackInfo,
    data: Vec<u8>,
    /// (moof start, moof size, mdat start, mdat size)
    fragments: Vec<(usize, usize, usize, usize)>,
}

impl Segment {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("cannot read segment {}", path.display()))?;
        Self::from_bytes(data)
    }

    pub fn from_bytes(data: Vec<u8>) -> Result<Self> {
        let boxes = top_level(&data)?;
        let mut ftyp = None;
        let mut moov = None;
        let mut fragments = Vec::new();
        let mut pending_moof: Option<(usize, usize)> = None;
        for (typ, off, _hl, size) in boxes {
            match &typ {
                b"ftyp" => ftyp = Some(data[off..off + size].to_vec()),
                b"moov" => moov = Some(parse_box(&data[off..off + size])?),
                b"moof" => pending_moof = Some((off, size)),
                b"mdat" => {
                    if let Some((mo, ms)) = pending_moof.take() {
                        fragments.push((mo, ms, off, size));
                    }
                }
                _ => {}
            }
        }
        let moov = moov.ok_or_else(|| anyhow!("segment has no moov"))?;
        let info = track_info(&moov)?;
        Ok(Self { ftyp: ftyp.ok_or_else(|| anyhow!("segment has no ftyp"))?, moov, info, data, fragments })
    }
}

/// One fragment rewritten for its new place in the master.
pub struct PlacedFragment {
    pub moof: Vec<u8>,
    pub base_time: u64,
    pub duration: u64,
}

/// Rewrites a moof: new sequence number, new decode time, track defaults made
/// explicit (so the master's trex does not matter) and data offsets fixed up.
fn place_fragment(moof_bytes: &[u8], info: &TrackInfo, master_track_id: u32, seq: u32, time_offset: i128) -> Result<PlacedFragment> {
    let mut moof = parse_box(moof_bytes)?;
    let old_len = moof_bytes.len() as i64;
    let children = moof.children_mut().ok_or_else(|| anyhow!("bad moof"))?;
    let mut base_time = 0u64;
    let mut duration = 0u64;
    let mut trafs = 0;
    for c in children.iter_mut() {
        match c.typ() {
            b"mfhd" => {
                let p = c.payload_mut().unwrap();
                p[4..8].copy_from_slice(&seq.to_be_bytes());
            }
            b"traf" => {
                trafs += 1;
                let tc = c.children_mut().unwrap();
                // tfhd
                let tfhd = tc.iter_mut().find(|b| b.typ() == b"tfhd").ok_or_else(|| anyhow!("traf without tfhd"))?;
                let p = tfhd.payload_mut().unwrap();
                let flags = be32(p, 0) & 0x00ff_ffff;
                if flags & 0x01 != 0 {
                    bail!("segment uses absolute data offsets (encode with default_base_moof)");
                }
                let mut fields = Vec::new();
                let mut o = 8;
                let sdi = if flags & 0x02 != 0 {
                    let v = be32(p, o);
                    o += 4;
                    Some(v)
                } else {
                    None
                };
                let dur = if flags & 0x08 != 0 {
                    let v = be32(p, o);
                    o += 4;
                    v
                } else {
                    info.trex_duration
                };
                let size = if flags & 0x10 != 0 {
                    let v = be32(p, o);
                    o += 4;
                    v
                } else {
                    info.trex_size
                };
                let sflags = if flags & 0x20 != 0 { be32(p, o) } else { info.trex_flags };
                let new_flags = (flags & !0x3f) | 0x08 | 0x10 | 0x20 | if sdi.is_some() { 0x02 } else { 0 } | 0x02_0000;
                fields.extend_from_slice(&new_flags.to_be_bytes());
                fields.extend_from_slice(&master_track_id.to_be_bytes());
                if let Some(s) = sdi {
                    fields.extend_from_slice(&s.to_be_bytes());
                }
                fields.extend_from_slice(&dur.to_be_bytes());
                fields.extend_from_slice(&size.to_be_bytes());
                fields.extend_from_slice(&sflags.to_be_bytes());
                *p = fields;
                // durations from trun
                for t in tc.iter().filter(|b| b.typ() == b"trun") {
                    let tp = t.payload().unwrap();
                    let tflags = be32(tp, 0) & 0x00ff_ffff;
                    let count = be32(tp, 4) as usize;
                    let mut o = 8;
                    if tflags & 0x01 != 0 {
                        o += 4;
                    }
                    if tflags & 0x04 != 0 {
                        o += 4;
                    }
                    let per = [0x100, 0x200, 0x400, 0x800].iter().filter(|&&f| tflags & f != 0).count() * 4;
                    if tflags & 0x100 != 0 {
                        for i in 0..count {
                            duration += be32(tp, o + i * per) as u64;
                        }
                    } else {
                        duration += dur as u64 * count as u64;
                    }
                }
                // tfdt -> version 1 with the new base time
                let orig = match tc.iter().find(|b| b.typ() == b"tfdt").and_then(Mp4Box::payload) {
                    Some(tp) if tp[0] == 1 => be64(tp, 4),
                    Some(tp) => be32(tp, 4) as u64,
                    None => 0,
                };
                let new_time = orig as i128 + time_offset;
                if new_time < 0 {
                    bail!("negative decode time");
                }
                base_time = new_time as u64;
                let mut tp = vec![1, 0, 0, 0];
                tp.extend_from_slice(&base_time.to_be_bytes());
                match tc.iter_mut().find(|b| b.typ() == b"tfdt") {
                    Some(b) => *b.payload_mut().unwrap() = tp,
                    None => {
                        let pos = tc.iter().position(|b| b.typ() == b"tfhd").unwrap() + 1;
                        tc.insert(pos, Mp4Box::Leaf { typ: *b"tfdt", payload: tp });
                    }
                }
            }
            _ => {}
        }
    }
    if trafs != 1 {
        bail!("expected exactly one traf per moof, found {trafs}");
    }
    // The moof changed size; trun data offsets are relative to the moof start
    // and point into the following mdat, so shift them by the difference.
    let delta = moof.encoded_len() as i64 - old_len;
    if delta != 0 {
        if let Some(traf) = moof.find_mut(&[b"traf"]).and_then(Mp4Box::children_mut) {
            for t in traf.iter_mut().filter(|b| b.typ() == b"trun") {
                let tp = t.payload_mut().unwrap();
                if be32(tp, 0) & 0x01 != 0 {
                    let v = i32::from_be_bytes(tp[8..12].try_into().unwrap()) as i64 + delta;
                    tp[8..12].copy_from_slice(&(v as i32).to_be_bytes());
                }
            }
        }
    }
    Ok(PlacedFragment { moof: moof.to_bytes(), base_time, duration })
}

/// mfra with one tfra entry per fragment: (time, moof offset).
pub fn build_mfra(track_id: u32, entries: &[(u64, u64)]) -> Vec<u8> {
    let mut tfra = vec![1, 0, 0, 0];
    tfra.extend_from_slice(&track_id.to_be_bytes());
    tfra.extend_from_slice(&0u32.to_be_bytes()); // 1-byte traf/trun/sample numbers
    tfra.extend_from_slice(&(entries.len() as u32).to_be_bytes());
    for (time, off) in entries {
        tfra.extend_from_slice(&time.to_be_bytes());
        tfra.extend_from_slice(&off.to_be_bytes());
        tfra.extend_from_slice(&[1, 1, 1]);
    }
    let tfra_len = 8 + tfra.len();
    let total = 8 + tfra_len + 16;
    let mfra = Mp4Box::Container {
        typ: *b"mfra",
        children: vec![
            Mp4Box::Leaf { typ: *b"tfra", payload: tfra },
            Mp4Box::Leaf { typ: *b"mfro", payload: [vec![0, 0, 0, 0], (total as u32).to_be_bytes().to_vec()].concat() },
        ],
    };
    mfra.to_bytes()
}

/// Durable description of the master after an operation.
#[derive(Clone, Debug)]
pub struct MasterLayout {
    pub data_end: u64,
    pub total_duration: u64,
    pub next_seq: u32,
    pub timescale: u32,
    pub movie_timescale: u32,
    pub track_id: u32,
    pub stsd: Vec<u8>,
    pub moov_offset: u64,
    pub moov_size: u64,
}

/// Compatibility of a segment with the master's sample description.
#[derive(Debug, PartialEq, Eq)]
pub enum Compat {
    Same,
    /// Same codec and size, different codec parameters: decoders pick up the
    /// in-band parameter sets that every keyframe carries.
    InBand,
    Incompatible(String),
}

pub fn compat(master: &MasterLayout, seg: &TrackInfo) -> Compat {
    if seg.timescale != master.timescale {
        return Compat::Incompatible(format!("timescale {} != {}", seg.timescale, master.timescale));
    }
    if seg.stsd == master.stsd {
        return Compat::Same;
    }
    // stsd: version/flags(4) entry_count(4) then entry: size(4) type(4) reserved(6) dri(2)
    // pre_defined(16) width(2) height(2)
    let key = |s: &[u8]| -> Option<(Vec<u8>, u16, u16)> {
        if s.len() < 44 {
            return None;
        }
        Some((s[12..16].to_vec(), u16::from_be_bytes([s[40], s[41]]), u16::from_be_bytes([s[42], s[43]])))
    };
    match (key(&master.stsd), key(&seg.stsd)) {
        (Some(a), Some(b)) if a == b => Compat::InBand,
        (a, b) => Compat::Incompatible(format!("sample entries differ ({a:?} vs {b:?})")),
    }
}

/// Creates a master file with the header of `first` and no fragments yet.
pub fn create_master(path: &Path, first: &Segment) -> Result<MasterLayout> {
    let mut moov = master_moov(&first.moov)?;
    let mut f = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    f.write_all(&first.ftyp)?;
    let moov_offset = first.ftyp.len() as u64;
    let moov_bytes = moov.to_bytes();
    f.write_all(&moov_bytes)?;
    drop(f);
    let layout = MasterLayout {
        data_end: moov_offset + moov_bytes.len() as u64,
        total_duration: 0,
        next_seq: 1,
        timescale: first.info.timescale,
        movie_timescale: first.info.movie_timescale,
        track_id: first.info.track_id,
        stsd: first.info.stsd.clone(),
        moov_offset,
        moov_size: moov_bytes.len() as u64,
    };
    finalize(path, &layout, &[], &mut moov)?;
    Ok(layout)
}

/// Creates a new master file containing the given segments (at least one).
#[cfg(test)]
pub fn write_new_master(path: &Path, segments: &[&Segment]) -> Result<(MasterLayout, Vec<FragmentEntry>)> {
    let first = segments.first().ok_or_else(|| anyhow!("no segments"))?;
    let mut layout = create_master(path, first)?;
    let mut index = Vec::new();
    let mut all_new = Vec::new();
    for s in segments {
        if let Compat::Incompatible(why) = compat(&layout, &s.info) {
            bail!("segment cannot be joined: {why}");
        }
        all_new.extend(append_segment(path, &mut layout, s, &mut index)?);
    }
    Ok((layout, all_new))
}

/// Appends one segment to an existing master. `index` holds all (time, offset)
/// pairs of the master (old + new are needed to rewrite the mfra).
pub fn append_segment(path: &Path, layout: &mut MasterLayout, seg: &Segment, index: &mut Vec<(u64, u64)>) -> Result<Vec<FragmentEntry>> {
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    let len = f.metadata()?.len();
    if len < layout.data_end {
        bail!("timelapse file is shorter than recorded in the index ({len} < {}); run `tapdream rebuild`", layout.data_end);
    }
    // Read and validate the header before touching anything.
    let mut moov_bytes = vec![0u8; layout.moov_size as usize];
    f.seek(SeekFrom::Start(layout.moov_offset))?;
    f.read_exact(&mut moov_bytes)?;
    let mut moov = parse_box(&moov_bytes)?;
    if moov.typ() != b"moov" {
        bail!("timelapse header is not where the index expects it; run `tapdream rebuild`");
    }
    drop(f);
    let new = append_fragments(path, layout, seg, index)?;
    finalize(path, layout, index, &mut moov)?;
    Ok(new)
}

fn append_fragments(path: &Path, layout: &mut MasterLayout, seg: &Segment, index: &mut Vec<(u64, u64)>) -> Result<Vec<FragmentEntry>> {
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    // Drop the old random-access index and anything a crash left behind.
    f.set_len(layout.data_end)?;
    f.seek(SeekFrom::Start(layout.data_end))?;
    let mut out = std::io::BufWriter::with_capacity(4 << 20, &mut f);
    let first_time = seg
        .fragments
        .first()
        .map(|&(mo, ms, _, _)| -> Result<u64> {
            let moof = parse_box(&seg.data[mo..mo + ms])?;
            Ok(match moof.find(&[b"traf", b"tfdt"]).and_then(Mp4Box::payload) {
                Some(p) if p[0] == 1 => be64(p, 4),
                Some(p) => be32(p, 4) as u64,
                None => 0,
            })
        })
        .transpose()?
        .unwrap_or(0);
    let offset = layout.total_duration as i128 - first_time as i128;
    let mut pos = layout.data_end;
    let mut new = Vec::new();
    for &(mo, ms, md, mds) in &seg.fragments {
        let placed = place_fragment(&seg.data[mo..mo + ms], &seg.info, layout.track_id, layout.next_seq, offset)?;
        out.write_all(&placed.moof)?;
        out.write_all(&seg.data[md..md + mds])?;
        index.push((placed.base_time, pos));
        new.push((layout.next_seq, placed.base_time, pos));
        pos += placed.moof.len() as u64 + mds as u64;
        layout.next_seq += 1;
        layout.total_duration = layout.total_duration.max(placed.base_time + placed.duration);
    }
    out.flush()?;
    drop(out);
    layout.data_end = pos;
    Ok(new)
}

fn finalize(path: &Path, layout: &MasterLayout, index: &[(u64, u64)], moov: &mut Mp4Box) -> Result<()> {
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.set_len(layout.data_end)?;
    f.seek(SeekFrom::Start(layout.data_end))?;
    f.write_all(&build_mfra(layout.track_id, index))?;
    set_master_duration(moov, layout.total_duration, layout.timescale, layout.movie_timescale)?;
    let bytes = moov.to_bytes();
    if bytes.len() as u64 != layout.moov_size {
        bail!("internal error: header size changed");
    }
    f.seek(SeekFrom::Start(layout.moov_offset))?;
    f.write_all(&bytes)?;
    f.sync_all()?;
    Ok(())
}

/// Removes anything after `data_end` that a crash may have left (then the
/// next append rewrites the index).
pub fn recover(path: &Path, layout: &MasterLayout, index: &[(u64, u64)]) -> Result<bool> {
    let len = std::fs::metadata(path)?.len();
    let expected = layout.data_end + build_mfra(layout.track_id, index).len() as u64;
    if len == expected {
        return Ok(false);
    }
    if len < layout.data_end {
        bail!("timelapse file was truncated; run `tapdream rebuild`");
    }
    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.set_len(layout.data_end)?;
    f.seek(SeekFrom::Start(layout.data_end))?;
    f.write_all(&build_mfra(layout.track_id, index))?;
    f.sync_all()?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn make_segment(dir: &Path, name: &str, frames: u32, color: &str) -> Option<std::path::PathBuf> {
        let out = dir.join(name);
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
            .arg(format!("color=c={color}:size=160x120:rate=30,drawtext=text=%{{n}}:fontsize=40:x=10:y=10,format=yuv420p"))
            .args(["-frames:v", &frames.to_string(), "-c:v", "libx264", "-g", "30", "-bf", "0", "-bsf:v", "dump_extra=freq=keyframe"])
            .args(["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-video_track_timescale", "90000", "-f", "mp4"])
            .arg(&out)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        // drawtext may be unavailable; retry without it
        if !ok {
            let ok = Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
                .arg(format!("color=c={color}:size=160x120:rate=30,format=yuv420p"))
                .args(["-frames:v", &frames.to_string(), "-c:v", "libx264", "-g", "30", "-bf", "0"])
                .args(["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-video_track_timescale", "90000", "-f", "mp4"])
                .arg(&out)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok {
                return None;
            }
        }
        Some(out)
    }

    fn probe(path: &Path) -> (u64, f64) {
        let o = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames:format=duration",
                "-of",
                "default=nw=1:nk=1",
            ])
            .arg(path)
            .output()
            .unwrap();
        let s = String::from_utf8_lossy(&o.stdout);
        let mut it = s.lines();
        let frames = it.next().unwrap_or("0").trim().parse().unwrap_or(0);
        let dur = it.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
        (frames, dur)
    }

    #[test]
    fn append_grows_a_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let Some(a) = make_segment(dir.path(), "a.mp4", 75, "red") else { return };
        let b = make_segment(dir.path(), "b.mp4", 40, "blue").unwrap();
        let c = make_segment(dir.path(), "c.mp4", 61, "green").unwrap();
        let master = dir.path().join("master.mp4");
        let sa = Segment::load(&a).unwrap();
        let (mut layout, new) = write_new_master(&master, &[&sa]).unwrap();
        let mut index: Vec<(u64, u64)> = new.iter().map(|&(_, t, o)| (t, o)).collect();
        assert_eq!(probe(&master).0, 75);
        let size_before = layout.data_end;
        for p in [&b, &c] {
            let s = Segment::load(p).unwrap();
            assert_eq!(compat(&layout, &s.info), Compat::Same);
            append_segment(&master, &mut layout, &s, &mut index).unwrap();
        }
        assert!(layout.data_end > size_before);
        let (frames, dur) = probe(&master);
        assert_eq!(frames, 176);
        assert!((dur - 176.0 / 30.0).abs() < 0.05, "duration {dur}");
        // decodes without errors
        let o = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&master).args(["-f", "null", "-"]).output().unwrap();
        assert!(o.stderr.is_empty(), "{}", String::from_utf8_lossy(&o.stderr));
        // a crash after writing fragments but before committing is undone
        let committed = layout.clone();
        let committed_index = index.clone();
        let s = Segment::load(&b).unwrap();
        append_segment(&master, &mut layout, &s, &mut index).unwrap();
        recover(&master, &committed, &committed_index).unwrap();
        // header still says 176 frames worth of time; the next append fixes it
        let mut layout = committed;
        let mut index = committed_index;
        append_segment(&master, &mut layout, &s, &mut index).unwrap();
        assert_eq!(probe(&master).0, 216);
        eprintln!("fmp4 append test ran: 216 frames");
        // the same content written in one go is byte-identical
        let master2 = dir.path().join("master2.mp4");
        let segs: Vec<Segment> = [&a, &b, &c, &b].iter().map(|p| Segment::load(p).unwrap()).collect();
        let refs: Vec<&Segment> = segs.iter().collect();
        write_new_master(&master2, &refs).unwrap();
        assert_eq!(std::fs::read(&master).unwrap(), std::fs::read(&master2).unwrap());
    }
}
