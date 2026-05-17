// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! Boot splash screen with ASCII seal art and progress bar.

use super::console::Console;
use super::framebuffer::Framebuffer;

const SEAL_COLOR: u32 = 0x0090B0D0;
const PROGRESS_BG: u32 = 0x00303040;
const PROGRESS_FG: u32 = 0x004488CC;
const VERSION_COLOR: u32 = 0x00808090;

pub fn render_splash(console: &mut Console) {
    console.clear();

    // Center the seal art
    for _ in 0..10 {
        console.write_str("\n");
    }

    console.write_colored("                          ___", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                       .-'   `'.", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                      /  .-=-.  \\", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                     | (  o o  ) |", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                      \\  `---'  /", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                  .----`-.___.'-`----.", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                 /                     \\", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                |   S E A L   O S   |", SEAL_COLOR);
    console.write_str("\n");
    console.write_colored("                 \\___________________/", SEAL_COLOR);
    console.write_str("\n\n");

    console.write_colored(
        "            The Geometrical Operating System",
        VERSION_COLOR,
    );
    console.write_str("\n");
    console.write_colored("                    v1.0.0-alpha", VERSION_COLOR);
    console.write_str("\n\n");
}

pub fn draw_progress_bar(fb: &Framebuffer, progress: u32) {
    if !fb.is_available() {
        return;
    }
    let bar_width = 400u32;
    let bar_height = 8u32;
    let bar_x = (fb.width - bar_width) / 2;
    let bar_y = fb.height - 100;

    // Background
    fb.fill_rect(bar_x, bar_y, bar_width, bar_height, PROGRESS_BG);

    // Fill
    let fill = (bar_width * progress.min(100)) / 100;
    fb.fill_rect(bar_x, bar_y, fill, bar_height, PROGRESS_FG);
}
