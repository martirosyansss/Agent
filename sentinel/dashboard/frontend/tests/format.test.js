import { describe, it, expect } from 'vitest';
import format from '@lib/format.js';

describe('format.formatPnl', () => {
    it('prefixes positive values with +$', () => {
        expect(format.formatPnl(12.34)).toBe('+$12.34');
    });
    it('uses -$ for negatives with absolute value', () => {
        expect(format.formatPnl(-5.5)).toBe('-$5.50');
    });
    it('renders zero as +$0.00', () => {
        expect(format.formatPnl(0)).toBe('+$0.00');
    });
    it('handles non-numeric input as 0', () => {
        expect(format.formatPnl('abc')).toBe('+$0.00');
        expect(format.formatPnl(null)).toBe('+$0.00');
        expect(format.formatPnl(undefined)).toBe('+$0.00');
    });
});

describe('format.formatUsd', () => {
    it('formats with 2 decimals and $ prefix', () => {
        expect(format.formatUsd(1234.5)).toBe('$1234.50');
    });
    it('null / undefined → $0.00', () => {
        expect(format.formatUsd(null)).toBe('$0.00');
        expect(format.formatUsd(undefined)).toBe('$0.00');
    });
});

describe('format.formatVol', () => {
    it('uses B suffix for billions', () => {
        expect(format.formatVol(2_500_000_000)).toBe('2.5B');
    });
    it('uses M suffix for millions', () => {
        expect(format.formatVol(1_250_000)).toBe('1.3M');
    });
    it('uses K suffix for thousands', () => {
        expect(format.formatVol(8_500)).toBe('8.5K');
    });
    it('no suffix for small numbers', () => {
        expect(format.formatVol(350)).toBe('350');
        expect(format.formatVol(0)).toBe('0');
    });
});

describe('format.pnlClass', () => {
    it('positive → "positive"', () => {
        expect(format.pnlClass(0.01)).toBe('positive');
    });
    it('negative → "negative"', () => {
        expect(format.pnlClass(-0.01)).toBe('negative');
    });
    it('zero → "neutral"', () => {
        expect(format.pnlClass(0)).toBe('neutral');
    });
    it('non-numeric → "neutral"', () => {
        expect(format.pnlClass('x')).toBe('neutral');
    });
});
