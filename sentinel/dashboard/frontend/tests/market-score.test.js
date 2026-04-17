import { describe, it, expect } from 'vitest';
import marketScore from '@lib/market-score.js';

const { scoreMarketDirection } = marketScore;

describe('market-score.scoreMarketDirection', () => {
    it('strong bullish setup → verdict up', () => {
        const r = scoreMarketDirection({
            trend: 'bullish',
            rsi_14: 55,
            macd_histogram: 1.2,
            close: 100,
            ema_50: 95,
            adx: 30,
            volume_ratio: 1.8,
            bb_upper: 105, bb_lower: 90,
        });
        expect(r.verdict).toBe('up');
        expect(r.score).toBeGreaterThanOrEqual(60);
        expect(r.reasons.length).toBeGreaterThan(0);
    });

    it('strong bearish setup → verdict down', () => {
        const r = scoreMarketDirection({
            trend: 'bearish',
            rsi_14: 75,              // overbought + bear trend
            macd_histogram: -1.5,
            close: 90,
            ema_50: 100,
            adx: 35,
            volume_ratio: 0.4,
            bb_upper: 100, bb_lower: 85,
        });
        expect(r.verdict).toBe('down');
        expect(r.score).toBeLessThanOrEqual(40);
    });

    it('neutral indicators → verdict flat', () => {
        const r = scoreMarketDirection({
            trend: 'neutral',
            rsi_14: 50,
            macd_histogram: 0,
            adx: 15,
            volume_ratio: 1.0,
        });
        expect(r.verdict).toBe('flat');
        expect(r.score).toBeGreaterThan(40);
        expect(r.score).toBeLessThan(60);
    });

    it('score is always clamped to [0, 100]', () => {
        const extreme = scoreMarketDirection({
            trend: 'bullish',
            rsi_14: 20,
            macd_histogram: 100,
            close: 100, ema_50: 50,
            adx: 60, volume_ratio: 5.0,
            bb_upper: 101, bb_lower: 99,
        });
        expect(extreme.score).toBeGreaterThanOrEqual(0);
        expect(extreme.score).toBeLessThanOrEqual(100);
    });

    it('reasons list is capped at 4 entries', () => {
        const r = scoreMarketDirection({
            trend: 'bullish',
            rsi_14: 25,
            macd_histogram: 5,
            close: 100, ema_50: 90,
            adx: 40, volume_ratio: 2,
            bb_upper: 110, bb_lower: 95,
        });
        expect(r.reasons.length).toBeLessThanOrEqual(4);
    });

    it('handles missing indicators gracefully', () => {
        const r = scoreMarketDirection({});
        expect(r).toHaveProperty('score');
        expect(r).toHaveProperty('verdict');
        expect(Array.isArray(r.reasons)).toBe(true);
    });

    it('handles null input', () => {
        const r = scoreMarketDirection(null);
        expect(r.score).toBe(50);
        expect(r.verdict).toBe('flat');
    });

    it('non-numeric RSI does not poison the score', () => {
        const r = scoreMarketDirection({ rsi_14: 'bogus', trend: 'neutral' });
        expect(Number.isFinite(r.score)).toBe(true);
    });
});
