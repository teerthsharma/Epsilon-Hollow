// Seal OS — Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: MIT

//! CMOS RTC driver (ports 0x70 / 0x71).

use core::sync::atomic::{AtomicU64, Ordering};
use spin::Mutex;
use x86_64::instructions::port::Port;

/// Real-time clock namespace.
pub struct Rtc;

/// Raw CMOS time registers.
#[derive(Clone, Copy, Debug)]
pub struct RtcTime {
    pub sec: u8,
    pub min: u8,
    pub hour: u8,
    pub day: u8,
    pub month: u8,
    pub year: u8,
    pub century: u8,
}

static INITIAL_EPOCH: AtomicU64 = AtomicU64::new(0);
static LAST_READ: Mutex<RtcTime> = Mutex::new(RtcTime {
    sec: 0,
    min: 0,
    hour: 0,
    day: 1,
    month: 1,
    year: 0,
    century: 20,
});

unsafe fn read_cmos(reg: u8) -> u8 {
    let mut index = Port::<u8>::new(0x70);
    let mut data = Port::<u8>::new(0x71);
    index.write(reg | 0x80);
    data.read()
}

unsafe fn write_cmos(reg: u8, val: u8) {
    let mut index = Port::<u8>::new(0x70);
    let mut data = Port::<u8>::new(0x71);
    index.write(reg | 0x80);
    data.write(val);
}

fn bcd_to_binary(val: u8) -> u8 {
    ((val >> 4) * 10) + (val & 0x0F)
}

fn binary_to_bcd(val: u8) -> u8 {
    ((val / 10) << 4) | (val % 10)
}

fn is_leap_year(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_month(year: u64, month: u64) -> u64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 30,
    }
}

fn days_since_epoch(year: u64, month: u64, day: u64) -> u64 {
    let mut days = 0u64;
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }
    for m in 1..month {
        days += days_in_month(year, m);
    }
    days += day.saturating_sub(1);
    days
}

/// Probe the RTC, read the current time, and cache the epoch.
pub fn init() {
    let time = read_time();
    let epoch = seconds_since_epoch_from(&time);
    INITIAL_EPOCH.store(epoch, Ordering::Relaxed);
    *LAST_READ.lock() = time;
}

/// Read the current time from CMOS registers.
pub fn read_time() -> RtcTime {
    let reg_b = unsafe { read_cmos(0x0B) };
    let binary = reg_b & 0x04 != 0;
    let hour_24 = reg_b & 0x02 != 0;

    let mut sec = unsafe { read_cmos(0x00) };
    let mut min = unsafe { read_cmos(0x02) };
    let mut hour = unsafe { read_cmos(0x04) };
    let _day_of_week = unsafe { read_cmos(0x06) };
    let mut day = unsafe { read_cmos(0x07) };
    let mut month = unsafe { read_cmos(0x08) };
    let mut year = unsafe { read_cmos(0x09) };
    let mut century = unsafe { read_cmos(0x32) };

    if !binary {
        sec = bcd_to_binary(sec);
        min = bcd_to_binary(min);
        hour = bcd_to_binary(hour);
        day = bcd_to_binary(day);
        month = bcd_to_binary(month);
        year = bcd_to_binary(year);
        century = bcd_to_binary(century);
    }

    if !hour_24 {
        let pm = hour & 0x80 != 0;
        hour = hour & 0x7F;
        if !binary {
            hour = bcd_to_binary(hour);
        }
        if hour == 12 {
            hour = if pm { 12 } else { 0 };
        } else if pm {
            hour += 12;
        }
    }

    let _ = _day_of_week;

    RtcTime {
        sec,
        min,
        hour,
        day,
        month,
        year,
        century,
    }
}

/// Write a time back to the CMOS registers.
///
/// Returns `true` on success.  This is a best-effort write; it does not
/// set the RTC "SET" bit and assumes the caller has verified the values.
pub fn set_time(time: &RtcTime) -> bool {
    let reg_b = unsafe { read_cmos(0x0B) };
    let binary = reg_b & 0x04 != 0;
    let hour_24 = reg_b & 0x02 != 0;

    let mut hour = time.hour;
    if !hour_24 {
        let mut pm = false;
        if hour >= 12 {
            pm = true;
            if hour > 12 {
                hour -= 12;
            }
        } else if hour == 0 {
            hour = 12;
        }
        if !binary {
            hour = binary_to_bcd(hour);
        }
        if pm {
            hour |= 0x80;
        }
    } else if !binary {
        hour = binary_to_bcd(hour);
    }

    let sec = if binary {
        time.sec
    } else {
        binary_to_bcd(time.sec)
    };
    let min = if binary {
        time.min
    } else {
        binary_to_bcd(time.min)
    };
    let day = if binary {
        time.day
    } else {
        binary_to_bcd(time.day)
    };
    let month = if binary {
        time.month
    } else {
        binary_to_bcd(time.month)
    };
    let year = if binary {
        time.year
    } else {
        binary_to_bcd(time.year)
    };
    let century = if binary {
        time.century
    } else {
        binary_to_bcd(time.century)
    };

    unsafe {
        write_cmos(0x00, sec);
        write_cmos(0x02, min);
        write_cmos(0x04, hour);
        write_cmos(0x07, day);
        write_cmos(0x08, month);
        write_cmos(0x09, year);
        write_cmos(0x32, century);
    }

    true
}

fn seconds_since_epoch_from(time: &RtcTime) -> u64 {
    let full_year = time.century as u64 * 100 + time.year as u64;
    let days = days_since_epoch(full_year, time.month as u64, time.day as u64);
    days * 86400 + time.hour as u64 * 3600 + time.min as u64 * 60 + time.sec as u64
}

/// Return the number of seconds since the Unix epoch (1970-01-01 00:00:00).
pub fn seconds_since_epoch() -> u64 {
    let time = read_time();
    seconds_since_epoch_from(&time)
}
