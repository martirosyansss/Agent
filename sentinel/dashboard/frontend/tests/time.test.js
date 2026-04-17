import { describe, it, expect } from 'vitest';
import time from '@lib/time.js';

describe('time.parseOpenedAt', () => {
    it('parses 13-digit Unix ms string', () => {
        const d = time.parseOpenedAt('1713180000000');
        expect(d).toBeInstanceOf(Date);
        expect(d.getTime()).toBe(1713180000000);
    });
    it('parses 13-digit Unix ms number', () => {
        const d = time.parseOpenedAt(1713180000000);
        expect(d.getTime()).toBe(1713180000000);
    });
    it('promotes 10-digit Unix seconds to ms (fixes 1970 bug)', () => {
        const d = time.parseOpenedAt('1713180000');
        expect(d.getTime()).toBe(1713180000 * 1000);
        expect(d.getFullYear()).toBeGreaterThan(2020);
    });
    it('parses ISO-8601 strings', () => {
        const d = time.parseOpenedAt('2024-04-15T10:40:00Z');
        expect(d).toBeInstanceOf(Date);
        expect(d.getUTCFullYear()).toBe(2024);
    });
    it('returns null for null / undefined / empty', () => {
        expect(time.parseOpenedAt(null)).toBeNull();
        expect(time.parseOpenedAt(undefined)).toBeNull();
        expect(time.parseOpenedAt('')).toBeNull();
    });
    it('returns null for gibberish', () => {
        expect(time.parseOpenedAt('not-a-date')).toBeNull();
    });
});

describe('time.formatDuration', () => {
    const now = 1_713_180_000_000;
    it('< 60s → "Xс"', () => {
        expect(time.formatDuration(now - 23_000, now)).toBe('23с');
    });
    it('< 1h → "Xм Yс"', () => {
        expect(time.formatDuration(now - (5 * 60 + 12) * 1000, now)).toBe('5м 12с');
    });
    it('< 24h → "Xч Yм"', () => {
        expect(time.formatDuration(now - (3 * 3600 + 5 * 60) * 1000, now)).toBe('3ч 5м');
    });
    it('days → "Xд Yч"', () => {
        expect(time.formatDuration(now - (2 * 86400 + 4 * 3600) * 1000, now)).toBe('2д 4ч');
    });
    it('negative diffs clamp to 0s', () => {
        expect(time.formatDuration(now + 10_000, now)).toBe('0с');
    });
    it('empty input → empty string', () => {
        expect(time.formatDuration(null, now)).toBe('');
    });
});

describe('time.timeAgoSeconds', () => {
    it('now for <60s', () => {
        expect(time.timeAgoSeconds(1000, 1010)).toBe('now');
    });
    it('minutes for <1h', () => {
        expect(time.timeAgoSeconds(0, 120)).toBe('2m');
    });
    it('hours for <1d', () => {
        expect(time.timeAgoSeconds(0, 7200)).toBe('2h');
    });
    it('days for 1d+', () => {
        expect(time.timeAgoSeconds(0, 3 * 86400)).toBe('3d');
    });
});
