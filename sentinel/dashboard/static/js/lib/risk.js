/**
 * SENTINEL — risk classification helpers.
 *
 * Maps a risk-usage percentage onto a traffic-light CSS class (and
 * companion colour class). Used by the risk panel + metric cards.
 */
(function (root, factory) {
    'use strict';
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.SENTINEL = root.SENTINEL || {};
        root.SENTINEL.risk = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    /** Return the progress-bar fill class for the given usage percentage. */
    function riskBarColor(pct) {
        var n = Number(pct) || 0;
        if (n < 40) return 'risk-bar-fill--green';
        if (n < 75) return 'risk-bar-fill--amber';
        return 'risk-bar-fill--red';
    }

    /** Return the text colour class for the given usage percentage. */
    function riskTextClass(pct) {
        var n = Number(pct) || 0;
        if (n >= 75) return 'risk-red';
        if (n >= 40) return 'risk-amber';
        return 'risk-green';
    }

    /**
     * Clamp a percentage into [0, 100].
     * Non-numeric / NaN input falls back to 0 rather than propagating NaN.
     */
    function clampPct(pct) {
        var n = Number(pct);
        if (!isFinite(n)) return 0;
        if (n < 0) return 0;
        if (n > 100) return 100;
        return n;
    }

    return {
        riskBarColor: riskBarColor,
        riskTextClass: riskTextClass,
        clampPct: clampPct,
    };
}));
