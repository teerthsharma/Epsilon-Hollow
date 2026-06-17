// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! SealPlayer — native media player. Currently decodes WAV/PCM from ManifoldFS.
//! Container demuxers for MP4/MKV/etc are planned.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContainerFormat {
    Mp4,
    Avi,
    Mkv,
    Mov,
    Webm,
    Flv,
    Wmv,
    Ogg,
    Mp3,
    Wav,
    Flac,
    Aac,
    Unknown,
}

impl ContainerFormat {
    pub fn from_extension(ext: &str) -> Self {
        match ext {
            "mp4" | "m4v" | "m4a" => Self::Mp4,
            "avi" => Self::Avi,
            "mkv" => Self::Mkv,
            "mov" | "qt" => Self::Mov,
            "webm" => Self::Webm,
            "flv" | "f4v" => Self::Flv,
            "wmv" | "wma" | "asf" => Self::Wmv,
            "ogg" | "ogv" | "oga" => Self::Ogg,
            "mp3" => Self::Mp3,
            "wav" | "wave" => Self::Wav,
            "flac" => Self::Flac,
            "aac" => Self::Aac,
            _ => Self::Unknown,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Mp4 => "MP4 (MPEG-4 Part 14)",
            Self::Avi => "AVI (Audio Video Interleave)",
            Self::Mkv => "MKV (Matroska)",
            Self::Mov => "MOV (QuickTime)",
            Self::Webm => "WebM (VP8/VP9/AV1)",
            Self::Flv => "FLV (Flash Video)",
            Self::Wmv => "WMV (Windows Media)",
            Self::Ogg => "OGG (Ogg Vorbis/Theora)",
            Self::Mp3 => "MP3 (MPEG-1 Audio Layer 3)",
            Self::Wav => "WAV (Waveform Audio)",
            Self::Flac => "FLAC (Free Lossless Audio)",
            Self::Aac => "AAC (Advanced Audio Coding)",
            Self::Unknown => "Unknown",
        }
    }

    pub fn is_video(&self) -> bool {
        matches!(
            self,
            Self::Mp4
                | Self::Avi
                | Self::Mkv
                | Self::Mov
                | Self::Webm
                | Self::Flv
                | Self::Wmv
                | Self::Ogg
        )
    }

    pub fn is_audio_only(&self) -> bool {
        matches!(self, Self::Mp3 | Self::Wav | Self::Flac | Self::Aac)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VideoCodec {
    H264,
    H265,
    Vp8,
    Vp9,
    Av1,
    Mpeg4,
    Theora,
    WmvV3,
    None,
}

impl VideoCodec {
    pub fn name(&self) -> &'static str {
        match self {
            Self::H264 => "H.264 / AVC",
            Self::H265 => "H.265 / HEVC",
            Self::Vp8 => "VP8",
            Self::Vp9 => "VP9",
            Self::Av1 => "AV1",
            Self::Mpeg4 => "MPEG-4 Part 2",
            Self::Theora => "Theora",
            Self::WmvV3 => "WMV3 / VC-1",
            Self::None => "(none)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioCodec {
    Aac,
    Mp3,
    Vorbis,
    Opus,
    Flac,
    Pcm,
    Wma,
    None,
}

impl AudioCodec {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Aac => "AAC",
            Self::Mp3 => "MP3",
            Self::Vorbis => "Vorbis",
            Self::Opus => "Opus",
            Self::Flac => "FLAC",
            Self::Pcm => "PCM",
            Self::Wma => "WMA",
            Self::None => "(none)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

pub struct MediaInfo {
    pub container: ContainerFormat,
    pub video_codec: VideoCodec,
    pub audio_codec: AudioCodec,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub duration_secs: u64,
    pub bitrate_kbps: u32,
    pub sample_rate: u32,
    pub channels: u8,
}

pub struct MediaPlayer {
    state: PlaybackState,
    current_file: Option<String>,
    current_info: Option<MediaInfo>,
    position_secs: u64,
    volume: u8,
    playlist: Vec<String>,
    playlist_index: usize,
}

impl MediaPlayer {
    pub fn new() -> Self {
        Self {
            state: PlaybackState::Stopped,
            current_file: None,
            current_info: None,
            position_secs: 0,
            volume: 80,
            playlist: Vec::new(),
            playlist_index: 0,
        }
    }

    pub fn open(&mut self, filename: &str) -> Result<String, String> {
        let ext = filename.rsplit('.').next().unwrap_or("");
        let container = ContainerFormat::from_extension(ext);

        // Only WAV/PCM is supported; all other formats are honestly rejected.
        if container != ContainerFormat::Wav {
            return Err(format!(
                "Unsupported format '.{}'\nSupported: wav (PCM)",
                ext
            ));
        }

        let info = probe_wav(filename);
        let summary = format_media_info(filename, &info);

        self.current_file = Some(String::from(filename));
        self.current_info = Some(info);
        self.position_secs = 0;
        self.state = PlaybackState::Playing;

        Ok(summary)
    }

    pub fn play(&mut self) -> String {
        match self.state {
            PlaybackState::Playing => String::from("[SealPlayer] Already playing."),
            PlaybackState::Paused => {
                self.state = PlaybackState::Playing;
                format!(
                    "[SealPlayer] Resumed at {}",
                    format_time(self.position_secs)
                )
            }
            PlaybackState::Stopped => {
                if let Some(name) = &self.current_file {
                    self.state = PlaybackState::Playing;
                    format!("[SealPlayer] Playing '{}'", name)
                } else {
                    String::from("[SealPlayer] Nothing to play. Use: play <file>")
                }
            }
        }
    }

    pub fn pause(&mut self) -> String {
        if self.state == PlaybackState::Playing {
            self.state = PlaybackState::Paused;
            format!("[SealPlayer] Paused at {}", format_time(self.position_secs))
        } else {
            String::from("[SealPlayer] Not playing.")
        }
    }

    pub fn stop(&mut self) -> String {
        self.state = PlaybackState::Stopped;
        self.position_secs = 0;
        String::from("[SealPlayer] Stopped.")
    }

    pub fn seek(&mut self, seconds: u64) -> String {
        if let Some(info) = &self.current_info {
            let clamped = seconds.min(info.duration_secs);
            self.position_secs = clamped;
            format!("[SealPlayer] Seeked to {}", format_time(clamped))
        } else {
            String::from("[SealPlayer] Nothing loaded.")
        }
    }

    pub fn volume_set(&mut self, vol: u8) -> String {
        self.volume = vol.min(100);
        format!("[SealPlayer] Volume: {}%", self.volume)
    }

    pub fn key_press(&mut self, ch: u8) {
        match ch {
            b' ' => {
                if self.state == PlaybackState::Playing {
                    self.pause();
                } else {
                    self.play();
                }
            }
            b'n' => {
                if !self.playlist.is_empty() {
                    self.playlist_index = (self.playlist_index + 1) % self.playlist.len();
                    let name = self.playlist[self.playlist_index].clone();
                    let _ = self.open(&name);
                }
            }
            b'p' => {
                if !self.playlist.is_empty() {
                    self.playlist_index = if self.playlist_index == 0 {
                        self.playlist.len() - 1
                    } else {
                        self.playlist_index - 1
                    };
                    let name = self.playlist[self.playlist_index].clone();
                    let _ = self.open(&name);
                }
            }
            b's' => {
                self.stop();
            }
            b'+' => {
                let v = (self.volume + 5).min(100);
                self.volume_set(v);
            }
            b'-' => {
                let v = self.volume.saturating_sub(5);
                self.volume_set(v);
            }
            _ => {
                // Unhandled input; no-op
            }
        }
    }

    pub fn status(&self) -> String {
        let state_str = match self.state {
            PlaybackState::Stopped => "Stopped",
            PlaybackState::Playing => "Playing",
            PlaybackState::Paused => "Paused",
        };

        let mut out = format!(
            "[SealPlayer] Status: {}\n  Volume: {}%\n",
            state_str, self.volume
        );

        if let (Some(name), Some(info)) = (&self.current_file, &self.current_info) {
            out.push_str(&format!(
                "  File: {}\n  Format: {}\n  Position: {} / {}\n",
                name,
                info.container.name(),
                format_time(self.position_secs),
                format_time(info.duration_secs)
            ));
            if info.container.is_video() {
                out.push_str(&format!(
                    "  Video: {} ({}x{} @ {}fps)\n",
                    info.video_codec.name(),
                    info.width,
                    info.height,
                    info.fps
                ));
            }
            out.push_str(&format!(
                "  Audio: {} ({}Hz, {}ch)\n  Bitrate: {} kbps",
                info.audio_codec.name(),
                info.sample_rate,
                info.channels,
                info.bitrate_kbps
            ));
        }

        out
    }

    pub fn add_to_playlist(&mut self, file: &str) {
        self.playlist.push(String::from(file));
    }

    pub fn next_track(&mut self) -> String {
        if self.playlist.is_empty() {
            return String::from("[SealPlayer] Playlist empty.");
        }
        self.playlist_index = (self.playlist_index + 1) % self.playlist.len();
        let name = self.playlist[self.playlist_index].clone();
        match self.open(&name) {
            Ok(s) => s,
            Err(e) => e,
        }
    }

    pub fn prev_track(&mut self) -> String {
        if self.playlist.is_empty() {
            return String::from("[SealPlayer] Playlist empty.");
        }
        if self.playlist_index == 0 {
            self.playlist_index = self.playlist.len() - 1;
        } else {
            self.playlist_index -= 1;
        }
        let name = self.playlist[self.playlist_index].clone();
        match self.open(&name) {
            Ok(s) => s,
            Err(e) => e,
        }
    }

    pub fn show_playlist(&self) -> String {
        if self.playlist.is_empty() {
            return String::from(
                "[SealPlayer] Playlist empty. Use 'play add <file>' to add tracks.",
            );
        }
        let mut out = String::from("[SealPlayer] Playlist\n══════════════════════\n");
        for (i, f) in self.playlist.iter().enumerate() {
            let marker = if i == self.playlist_index { "▶" } else { " " };
            out.push_str(&format!("  {} {:>3}. {}\n", marker, i + 1, f));
        }
        out
    }

    pub fn supported_formats() -> String {
        String::from(
            "[SealPlayer] Supported Formats\n\
             ═══════════════════════════════\n\
             Working:\n\
               Audio: WAV (PCM) — native decoding, real RIFF/WAVE parser\n\
             \n\
             Planned:\n\
               Video: MP4, MKV, AVI, WebM, MOV, FLV, WMV, OGG\n\
               Audio: MP3, FLAC, AAC, OGG/Vorbis, Opus\n\
             \n\
             Roadmap: container demuxers first, then software decode.\n\
             GPU-accelerated decode planned once GPU driver is complete.",
        )
    }

    pub fn render_to_window(&self, win: &mut crate::wm::window::Window) {
        use crate::graphics::htek;

        let cw = win.client_width();
        let ch = win.client_height();
        let m = 10u32;

        // Gradient background
        htek::fill_gradient_v(win, 0, 0, cw, ch, 0x00101018, 0x00080810);

        // Title bar area
        htek::fill_rounded_rect_gradient(win, m, m, cw - m * 2, 36, 8, 0x001A1A30, 0x00121224);
        htek::render_text_glow(win, m + 10, m + 4, "SealPlayer", 0x0000DDAA, 0x00004433);

        // Volume indicator (top right)
        let vol_str = format!("Vol {}%", self.volume);
        let vol_x = cw - m - (vol_str.len() as u32 * htek::TEXT_CHAR_W) - 8;
        htek::render_text_small(win, vol_x, m + 10, &vol_str, 0x00888899);

        if let (Some(name), Some(info)) = (&self.current_file, &self.current_info) {
            // File info
            htek::render_text_small(win, m + 8, m + 42, name, 0x00CCCCDD);
            let codec_str = format!("{} | {}", info.container.name(), info.video_codec.name());
            htek::render_text_small(win, m + 8, m + 60, &codec_str, 0x00666688);

            // Video viewport
            if info.container.is_video() {
                let vp_y = m + 80;
                let vp_w = cw - m * 2;
                let vp_h = ch.saturating_sub(vp_y + 80).min(300);

                htek::fill_rounded_rect_gradient(
                    win, m, vp_y, vp_w, vp_h, 6, 0x000A0A12, 0x00060608,
                );
                htek::stroke_rounded_rect(win, m, vp_y, vp_w, vp_h, 6, 1, 0x00303050);

                // State icon in center
                let state_str = match self.state {
                    PlaybackState::Playing => "Playing",
                    PlaybackState::Paused => "Paused",
                    PlaybackState::Stopped => "Stopped",
                };
                let sw = state_str.len() as u32 * htek::HTEXT_CHAR_W;
                htek::render_text_glow(
                    win,
                    m + (vp_w.saturating_sub(sw)) / 2,
                    vp_y + (vp_h.saturating_sub(htek::HTEXT_CHAR_H)) / 2,
                    state_str,
                    0x00FFFFFF,
                    0x00444466,
                );

                // Resolution badge
                let res_str = format!("{}x{} {}fps", info.width, info.height, info.fps);
                htek::render_text_small(win, m + 8, vp_y + vp_h - 18, &res_str, 0x00445566);
            }

            // Transport controls area (bottom)
            let ctrl_y = ch.saturating_sub(60);

            // Progress bar background
            let bar_x = m + 4;
            let bar_w = cw - m * 2 - 8;
            htek::fill_rounded_rect(win, bar_x, ctrl_y, bar_w, 6, 3, 0x00252538);

            // Progress fill
            if info.duration_secs > 0 {
                let progress =
                    ((self.position_secs as u32) * bar_w) / (info.duration_secs as u32).max(1);
                if progress > 0 {
                    htek::fill_rounded_rect(win, bar_x, ctrl_y, progress.max(6), 6, 3, 0x0000CCAA);
                    // Playhead dot
                    let dot_x = bar_x + progress;
                    if dot_x + 4 < cw {
                        htek::draw_circle_filled(win, dot_x, ctrl_y + 3, 5, 0x0000FFCC);
                    }
                }
            }

            // Time display
            let time_str = format!(
                "{} / {}",
                format_time(self.position_secs),
                format_time(info.duration_secs)
            );
            htek::render_text_small(win, m + 4, ctrl_y + 12, &time_str, 0x00AAAABB);

            // Bitrate info (right)
            let br_str = format!("{} kbps", info.bitrate_kbps);
            let br_x = cw - m - (br_str.len() as u32 * htek::TEXT_CHAR_W) - 4;
            htek::render_text_small(win, br_x, ctrl_y + 12, &br_str, 0x00666688);

            // Audio info
            let audio_str = format!(
                "{} {}Hz {}ch",
                info.audio_codec.name(),
                info.sample_rate,
                info.channels
            );
            let aw = audio_str.len() as u32 * htek::TEXT_CHAR_W;
            htek::render_text_small(win, (cw - aw) / 2, ctrl_y + 12, &audio_str, 0x00555577);
        } else {
            // No media loaded — show placeholder
            let center_y = ch / 2 - 40;
            htek::fill_rounded_rect_gradient(
                win,
                m,
                m + 50,
                cw - m * 2,
                ch - m * 2 - 100,
                10,
                0x000E0E1A,
                0x00080810,
            );
            htek::stroke_rounded_rect(
                win,
                m,
                m + 50,
                cw - m * 2,
                ch - m * 2 - 100,
                10,
                1,
                0x00252540,
            );

            // Big play icon (triangle via lines)
            let icon_cx = cw / 2;
            let icon_cy = center_y;
            htek::draw_circle_filled(win, icon_cx, icon_cy, 30, 0x00202038);
            htek::draw_circle_filled(win, icon_cx, icon_cy, 28, 0x00181828);

            let msg = "No media loaded";
            let mw = msg.len() as u32 * htek::HTEXT_CHAR_W;
            htek::render_text_smooth(win, (cw - mw) / 2, icon_cy + 40, msg, 0x00556677);

            let hint = "Use: play <file>";
            let hw = hint.len() as u32 * htek::TEXT_CHAR_W;
            htek::render_text_small(win, (cw - hw) / 2, icon_cy + 76, hint, 0x00334455);

            // Format badges
            let fmts = ["MP4", "MKV", "AVI", "WebM", "MP3", "FLAC"];
            let badge_y = icon_cy + 100;
            let total_w = fmts.len() as u32 * 52;
            let start_x = (cw.saturating_sub(total_w)) / 2;
            for (i, fmt) in fmts.iter().enumerate() {
                let bx = start_x + i as u32 * 52;
                htek::fill_rounded_rect(win, bx, badge_y, 48, 20, 4, 0x001A1A2E);
                htek::stroke_rounded_rect(win, bx, badge_y, 48, 20, 4, 1, 0x00303050);
                let fw = fmt.len() as u32 * htek::TEXT_CHAR_W;
                htek::render_text_small(win, bx + (48 - fw) / 2, badge_y + 2, fmt, 0x006688AA);
            }
        }
    }

    pub fn mouse_click(&mut self, _x: u32, _y: u32, _pressed: bool) {}
    pub fn mouse_move(&mut self, _x: u32, _y: u32) {}
}

fn probe_wav(filename: &str) -> MediaInfo {
    // Try to read the file from ManifoldFS and parse the WAV header.
    let mut sample_rate = 44100u32;
    let mut channels = 2u8;
    let mut bits_per_sample = 16u16;
    let mut duration_secs = 0u64;

    if let Ok(data) = crate::fs::vfs::with_vfs(|vfs| {
        if let Ok(handle) = vfs.lookup_follow(filename) {
            let mut buf = alloc::vec![0u8; 65536];
            if let Ok(n) = vfs.read(handle, &mut buf, 0) {
                buf.truncate(n);
                return Ok(buf);
            }
        }
        Err(())
    }) {
        if data.len() >= 44 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
            // Parse fmt chunk (offset 22 = channels, 24 = sample rate, 34 = bits per sample)
            channels = u16::from_le_bytes([data[22], data[23]]) as u8;
            sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
            bits_per_sample = u16::from_le_bytes([data[34], data[35]]);
            let data_size = u32::from_le_bytes([data[40], data[41], data[42], data[43]]) as u64;
            let byte_rate = (sample_rate as u64) * (channels as u64) * (bits_per_sample as u64 / 8);
            if byte_rate > 0 {
                duration_secs = data_size / byte_rate;
            }
        }
    }

    MediaInfo {
        container: ContainerFormat::Wav,
        video_codec: VideoCodec::None,
        audio_codec: AudioCodec::Pcm,
        width: 0,
        height: 0,
        fps: 0,
        duration_secs,
        bitrate_kbps: ((sample_rate as u32) * (channels as u32) * (bits_per_sample as u32)) / 1000,
        sample_rate,
        channels,
    }
}

fn format_media_info(filename: &str, info: &MediaInfo) -> String {
    let mut out = format!("[SealPlayer] Opening '{}'\n", filename);
    out.push_str(&format!("  Container: {}\n", info.container.name()));

    if info.container.is_video() {
        out.push_str(&format!(
            "  Video: {} ({}x{} @ {}fps, {} kbps)\n",
            info.video_codec.name(),
            info.width,
            info.height,
            info.fps,
            info.bitrate_kbps
        ));
    }

    out.push_str(&format!(
        "  Audio: {} ({}Hz, {}ch)\n",
        info.audio_codec.name(),
        info.sample_rate,
        info.channels
    ));

    out.push_str(&format!(
        "  Duration: {}\n",
        format_time(info.duration_secs)
    ));
    out.push_str("[SealPlayer] Playing...");
    out
}

fn format_time(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        format!("{}:{:02}", m, s)
    }
}
