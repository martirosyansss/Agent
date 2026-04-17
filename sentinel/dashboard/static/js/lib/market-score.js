/**
 * SENTINEL — market direction scoring.
 *
 * Combines EMA trend, RSI, MACD histogram, ADX, volume ratio, Bollinger
 * band position and price-vs-EMA50 into a single [0, 100] score with a
 * verdict (up / down / flat) and a capped list of human-readable reasons.
 *
 * Kept pure and deterministic so it can be regression-tested — the
 * dashboard renders the result directly, so scoring drift would change
 * what users see without any code review signal.
 */
(function (root, factory) {
    'use strict';
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.SENTINEL = root.SENTINEL || {};
        root.SENTINEL.marketScore = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    function _num(v, fallback) {
        var n = Number(v);
        if (!isFinite(n)) return (fallback == null ? 0 : fallback);
        return n;
    }

    /**
     * Compute a direction score for a symbol.
     *
     * @param {object} ind  indicator bundle (trend, rsi_14, macd_histogram,
     *                      close, ema_50, adx, volume_ratio, bb_upper, bb_lower)
     * @returns {{score:number, verdict:'up'|'down'|'flat', reasons:Array<{sign:string,text:string}>}}
     */
    function scoreMarketDirection(ind) {
        var o = ind || {};
        var score = 50;
        var reasons = [];

        // Trend (EMA9 vs EMA21) — primary driver
        if (o.trend === 'bullish') {
            score += 15;
            reasons.push({ sign: 'bull', text: 'EMA9 > EMA21 — восходящий импульс' });
        } else if (o.trend === 'bearish') {
            score -= 15;
            reasons.push({ sign: 'bear', text: 'EMA9 < EMA21 — нисходящий импульс' });
        }

        // RSI zones
        var rsi = _num(o.rsi_14, 50);
        if (rsi > 70) {
            score -= 12;
            reasons.push({ sign: 'bear', text: 'RSI ' + rsi.toFixed(1) + ' — перекупленность, риск отката' });
        } else if (rsi > 65) {
            score -= 4;
        } else if (rsi < 30) {
            score += 10;
            reasons.push({ sign: 'bull', text: 'RSI ' + rsi.toFixed(1) + ' — перепроданность, отскок вероятен' });
        } else if (rsi < 45) {
            score += 5;
        }

        // MACD histogram
        var hist = _num(o.macd_histogram, 0);
        if (hist > 0) {
            score += 10;
            reasons.push({ sign: 'bull', text: 'MACD гистограмма > 0 — импульс растёт' });
        } else if (hist < 0) {
            score -= 10;
            reasons.push({ sign: 'bear', text: 'MACD гистограмма < 0 — импульс падает' });
        }

        // Price vs EMA50 (global structure)
        var close = _num(o.close, 0);
        var ema50 = _num(o.ema_50, 0);
        if (ema50 > 0 && close > 0) {
            if (close > ema50) {
                score += 8;
                reasons.push({ sign: 'bull', text: 'Цена выше EMA50 — среднесрочная структура бычья' });
            } else {
                score -= 8;
                reasons.push({ sign: 'bear', text: 'Цена ниже EMA50 — структура медвежья' });
            }
        }

        // ADX — trend strength amplifier
        var adx = _num(o.adx, 0);
        if (adx >= 40) {
            if (o.trend === 'bullish') { score += 6; reasons.push({ sign: 'bull', text: 'ADX ' + adx.toFixed(1) + ' — очень сильный тренд' }); }
            else if (o.trend === 'bearish') { score -= 6; reasons.push({ sign: 'bear', text: 'ADX ' + adx.toFixed(1) + ' — очень сильный тренд' }); }
        } else if (adx >= 25) {
            if (o.trend === 'bullish') score += 4;
            else if (o.trend === 'bearish') score -= 4;
        } else {
            reasons.push({ sign: 'flat', text: 'ADX ' + adx.toFixed(1) + ' — слабый тренд, возможен флэт' });
        }

        // Volume confirmation
        var vol = _num(o.volume_ratio, 0);
        if (vol >= 1.5) {
            score += 4;
            reasons.push({ sign: 'bull', text: 'Объём x' + vol.toFixed(2) + ' — активный интерес' });
        } else if (vol > 0 && vol < 0.7) {
            score -= 2;
        }

        // Bollinger band position
        var bbU = _num(o.bb_upper, 0);
        var bbL = _num(o.bb_lower, 0);
        if (bbU > bbL && close > 0) {
            var bbPos = (close - bbL) / (bbU - bbL);
            if (bbPos > 0.95) {
                score -= 5;
                reasons.push({ sign: 'bear', text: 'Цена у верхней BB — сопротивление рядом' });
            } else if (bbPos < 0.05) {
                score += 5;
                reasons.push({ sign: 'bull', text: 'Цена у нижней BB — поддержка рядом' });
            }
        }

        if (score < 0) score = 0;
        if (score > 100) score = 100;

        var verdict = 'flat';
        if (score >= 60) verdict = 'up';
        else if (score <= 40) verdict = 'down';

        return {
            score: Math.round(score),
            verdict: verdict,
            reasons: reasons.slice(0, 4),
        };
    }

    return { scoreMarketDirection: scoreMarketDirection };
}));
