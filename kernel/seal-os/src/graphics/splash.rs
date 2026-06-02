// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Boot splash screen with ASCII seal art and progress bar.

use super::console::Console;
use super::framebuffer::Framebuffer;

pub fn render_splash(console: &mut Console) {
    console.clear();
    let theme = crate::wm::themes::current_theme();

    // Center the seal art
    for _ in 0..10 {
        console.write_str("\n");
    }

    console.write_colored("                          ___", theme.splash_color);
    console.write_str("\n");
    console.write_colored("                       .-'   `'.", theme.splash_color);
    console.write_str("\n");
    console.write_colored("                      /  .-=-.  \\", theme.splash_color);
    console.write_str("\n");
    console.write_colored("                     | (  o o  ) |", theme.splash_color);
    console.write_str("\n");
    console.write_colored("                      \\  `---'  /", theme.splash_color);
    console.write_str("\n");
    console.write_colored("                  .----`-.___.'-`----.", theme.splash_color);
    console.write_str("\n");
    console.write_colored(
        "                 /                     \\",
        theme.splash_color,
    );
    console.write_str("\n");
    console.write_colored("                |   S E A L   O S   |", theme.splash_color);
    console.write_str("\n");
    console.write_colored(
        "                 \\___________________/",
        theme.splash_color,
    );
    console.write_str("\n\n");

    console.write_colored("            The Geometrical Operating System", theme.fg);
    console.write_str("\n");
    console.write_colored("                    v0.4.6", theme.fg);
    console.write_str("\n\n");
}

pub fn draw_progress_bar(fb: &Framebuffer, progress: u32, label: &str) {
    if !fb.is_available() {
        return;
    }
    let theme = crate::wm::themes::current_theme();
    let bar_width = 400u32;
    let bar_height = 12u32;
    let bar_x = (fb.width - bar_width) / 2;
    let bar_y = fb.height - 120;

    // Subtle border
    fb.fill_rect(
        bar_x - 1,
        bar_y - 1,
        bar_width + 2,
        bar_height + 2,
        theme.border,
    );

    // Background
    fb.fill_rect(bar_x, bar_y, bar_width, bar_height, theme.progress_bg);

    // Fill
    let fill = (bar_width * progress.min(100)) / 100;
    fb.fill_rect(bar_x, bar_y, fill, bar_height, theme.progress_fg);

    // Label below the bar
    let label_y = bar_y + bar_height + 8;
    let label_w = (label.len() as u32) * super::font::CHAR_WIDTH;
    let label_x = if label.is_empty() {
        bar_x
    } else {
        (fb.width.saturating_sub(label_w)) / 2
    };

    // Clear old label text area to avoid ghosting
    fb.fill_rect(
        bar_x - 1,
        label_y,
        bar_width + 2,
        super::font::CHAR_HEIGHT + 2,
        theme.bg,
    );

    // Draw new label
    for (i, ch) in label.bytes().enumerate() {
        super::font::draw_char(
            fb,
            label_x + (i as u32) * super::font::CHAR_WIDTH,
            label_y,
            ch,
            theme.fg,
        );
    }
}
