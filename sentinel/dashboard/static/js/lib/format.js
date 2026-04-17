/**
 * SENTINEL — formatting helpers.
 *
 * Pure functions, side-effect free, so they can be unit-tested under
 * Node (Vitest) while still being loaded as a classic script in the
 * browser via UMD-style wrapper.
 */
(function (root, factory) {
    'use strict';
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.SENTINEL = root.SENTINEL || {};
        root.SENTINEL.format = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    /** Format a signed PnL amount with currency prefix. */
    function formatPnl(val) {
        var n = Number(val) || 0;
        if (n >= 0) return '+$' + n.toFixed(2);
        return '-$' + Math.abs(n).toFixed(2);
    }

    /** Format a USD amount with 2 decimal places. */
    function formatUsd(val) {
        return '$' + Number(val || 0).toFixed(2);
    }

    /**
     * Humanise a volume number.
     *   1_250_000 -> '1.3M'
     *   8_500     -> '8.5K'
     *   350       -> '350'
     */
    function formatVol(v) {
        var n = Number(v) || 0;
        if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return n.toFixed(0);
    }

    /** Map a PnL value to a CSS class name. */
    function pnlClass(val) {
        var n = Number(val) || 0;
        if (n > 0) return 'positive';
        if (n < 0) return 'negative';
        return 'neutral';
    }

    return { formatPnl: formatPnl, formatUsd: formatUsd, formatVol: formatVol, pnlClass: pnlClass };
}));
