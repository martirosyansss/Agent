import { describe, it, expect } from 'vitest';
import risk from '@lib/risk.js';

describe('risk.riskBarColor', () => {
    it('under 40% → green', () => {
        expect(risk.riskBarColor(0)).toBe('risk-bar-fill--green');
        expect(risk.riskBarColor(39.99)).toBe('risk-bar-fill--green');
    });
    it('40–74% → amber', () => {
        expect(risk.riskBarColor(40)).toBe('risk-bar-fill--amber');
        expect(risk.riskBarColor(74.99)).toBe('risk-bar-fill--amber');
    });
    it('75%+ → red', () => {
        expect(risk.riskBarColor(75)).toBe('risk-bar-fill--red');
        expect(risk.riskBarColor(100)).toBe('risk-bar-fill--red');
        expect(risk.riskBarColor(1000)).toBe('risk-bar-fill--red');
    });
    it('invalid input defaults to green (0)', () => {
        expect(risk.riskBarColor('x')).toBe('risk-bar-fill--green');
        expect(risk.riskBarColor(null)).toBe('risk-bar-fill--green');
    });
});

describe('risk.riskTextClass', () => {
    it('mirrors the thresholds used by the bar', () => {
        expect(risk.riskTextClass(0)).toBe('risk-green');
        expect(risk.riskTextClass(39.9)).toBe('risk-green');
        expect(risk.riskTextClass(40)).toBe('risk-amber');
        expect(risk.riskTextClass(74.9)).toBe('risk-amber');
        expect(risk.riskTextClass(75)).toBe('risk-red');
        expect(risk.riskTextClass(100)).toBe('risk-red');
    });
});

describe('risk.clampPct', () => {
    it('clamps below 0 to 0', () => {
        expect(risk.clampPct(-10)).toBe(0);
    });
    it('clamps above 100 to 100', () => {
        expect(risk.clampPct(250)).toBe(100);
    });
    it('passes valid values through', () => {
        expect(risk.clampPct(42)).toBe(42);
    });
    it('NaN / Infinity → 0', () => {
        expect(risk.clampPct(NaN)).toBe(0);
        expect(risk.clampPct(Infinity)).toBe(0);
        expect(risk.clampPct(-Infinity)).toBe(0);
    });
});
