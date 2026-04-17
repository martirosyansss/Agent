/**
 * SENTINEL — time helpers.
 *
 * Handles the trade-opened-at values we receive from the backend, which
 * can arrive as:
 *   • Unix ms (13-digit string or number)  e.g. "1713180000000"
 *   • Unix seconds (10-digit)              e.g. 1713180000
 *   • ISO-8601 / any Date-parseable string e.g. "2024-04-15T10:40:00Z"
 *
 * Everything returns millisecond precision so downstream code can diff
 * against Date.now() uniformly.
 */
(function (root, factory) {
    'use strict';
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.SENTINEL = root.SENTINEL || {};
        root.SENTINEL.time = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    /** Parse a trade-opened-at value into a Date, or null if unrecognisable. */
    function parseOpenedAt(val) {
        if (val == null || val === '') return null;
        var s = String(val);
        if (/^\d{13}$/.test(s)) return new Date(Number(s));
        if (/^\d{10}$/.test(s)) return new Date(Number(s) * 1000);
        var d = new Date(s);
        return isNaN(d.getTime()) ? null : d;
    }

    /**
     * Humanise an "opened-at" value as elapsed time.
     *   < 60s  -> "23с"
     *   < 1h   -> "5м 12с"
     *   < 24h  -> "3ч 5м"
     *   else   -> "2д 4ч"
     *
     * @param {*} val   opened-at raw value (see parseOpenedAt)
     * @param {number} [now=Date.now()]  override for deterministic tests
     */
    function formatDuration(val, now) {
        var opened = parseOpenedAt(val);
        if (!opened) return '';
        var ref = (typeof now === 'number') ? now : Date.now();
        var diff = Math.max(0, Math.floor((ref - opened.getTime()) / 1000));
        if (diff < 60) return diff + 'с';
        if (diff < 3600) return Math.floor(diff / 60) + 'м ' + (diff % 60) + 'с';
        var h = Math.floor(diff / 3600);
        var m = Math.floor((diff % 3600) / 60);
        if (h < 24) return h + 'ч ' + m + 'м';
        var d = Math.floor(h / 24);
        return d + 'д ' + (h % 24) + 'ч';
    }

    /**
     * Compact "time ago" for short labels.
     * Takes unix seconds. Returns 'now' / '3m' / '2h' / '1d'.
     */
    function timeAgoSeconds(ts, nowSeconds) {
        var ref = (typeof nowSeconds === 'number') ? nowSeconds : Math.floor(Date.now() / 1000);
        var diff = ref - Number(ts || 0);
        if (diff < 60) return 'now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h';
        return Math.floor(diff / 86400) + 'd';
    }

    return {
        parseOpenedAt: parseOpenedAt,
        formatDuration: formatDuration,
        timeAgoSeconds: timeAgoSeconds,
    };
}));
