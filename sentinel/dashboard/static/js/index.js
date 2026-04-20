
    'use strict';

    /* ══════════════════════════════════════════════════
       Observability: structured logging + global errors.
       All previously-silent `catch(e){}` blocks forward to
       _logError so failures surface during development and
       can be wired to a client-side telemetry endpoint later.
       ══════════════════════════════════════════════════ */
    /* Throttle: coalesce bursts of identical errors to avoid flooding the
       server (e.g. a render loop crashing on every WS tick). */
    var _lastErrSig = '';
    var _lastErrAt = 0;
    function _logError(scope, err, extra) {
        var payload = {
            ts: new Date().toISOString(),
            scope: String(scope || 'unknown'),
            msg: (err && err.message) || String(err || ''),
            stack: (err && err.stack) || null,
            extra: extra || null,
        };
        /* eslint-disable-next-line no-console */
        console.error('[SENTINEL]', payload);

        var sig = payload.scope + '|' + payload.msg;
        var now = Date.now();
        if (sig === _lastErrSig && (now - _lastErrAt) < 5000) return;
        _lastErrSig = sig;
        _lastErrAt = now;

        /* Fire-and-forget: use _csrfFetch if available (defined below) */
        try {
            var fn = (typeof _csrfFetch === 'function') ? _csrfFetch : fetch;
            fn('/api/client-error', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'same-origin',
                body: JSON.stringify({
                    scope: payload.scope,
                    msg: payload.msg,
                    extra: { stack: payload.stack ? payload.stack.slice(0, 2000) : null, extra: payload.extra },
                }),
                /* Don't keep the tab alive for this beacon */
                keepalive: true,
            }).catch(function() { /* swallow — can't log-the-logger */ });
        } catch (_) { /* swallow */ }
    }
    window.addEventListener('error', function(ev) {
        _logError('window.error', ev.error || ev.message, { file: ev.filename, line: ev.lineno, col: ev.colno });
    });
    window.addEventListener('unhandledrejection', function(ev) {
        _logError('unhandledrejection', ev.reason);
    });

    /* ── CSRF helper ──────────────────────────────
       Reads the `sentinel_csrf` cookie set by the backend on HTML loads
       and echoes it in a custom header for every mutating request.
       Required by CsrfMiddleware in app.py.
       ────────────────────────────────────────── */
    function _getCsrfToken() {
        var m = document.cookie.match(/(?:^|;\s*)sentinel_csrf=([^;]+)/);
        return m ? decodeURIComponent(m[1]) : '';
    }
    /* Dedup toast shown on 401/403 so rapid failures don't stack notifications. */
    var _lastAuthToastAt = 0;
    function _authToast(msg) {
        var now = Date.now();
        if (now - _lastAuthToastAt < 3000) return;
        _lastAuthToastAt = now;
        if (typeof showToast === 'function') showToast(msg, 'error');
    }

    function _csrfFetch(url, opts) {
        opts = opts || {};
        var method = (opts.method || 'GET').toUpperCase();
        if (method !== 'GET' && method !== 'HEAD') {
            var headers = new Headers(opts.headers || {});
            var tok = _getCsrfToken();
            if (tok) headers.set('X-CSRF-Token', tok);
            opts.headers = headers;
            opts.credentials = opts.credentials || 'same-origin';
        }
        return fetch(url, opts).then(function(r) {
            /* 401 = auth cookie expired/missing — send user to /login so they can recover.
               Skip the redirect for the beacon endpoint to avoid bouncing during tab unload. */
            if (r.status === 401 && url !== '/api/client-error') {
                _authToast('Требуется вход — перенаправляем…');
                setTimeout(function() { window.location.href = '/login'; }, 600);
            } else if (r.status === 403 && url !== '/api/client-error') {
                /* CSRF token mismatch — typically means the csrf cookie expired */
                _authToast('Сессия истекла — перезагрузите страницу (F5)');
            }
            return r;
        });
    }

    /* ── Helpers ──────────────────────────────── */
    /* Pure formatters come from static/js/lib/format.js (unit-tested).
       Bound here as plain vars so legacy call sites keep working unchanged. */
    var _FMT = (typeof SENTINEL !== 'undefined' && SENTINEL.format) || {};
    var _RISK = (typeof SENTINEL !== 'undefined' && SENTINEL.risk) || {};
    var formatPnl = _FMT.formatPnl;
    var formatUsd = _FMT.formatUsd;
    var formatVol = _FMT.formatVol;
    var pnlClass = _FMT.pnlClass;

    function escapeHtml(str) {
        var div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function toneClass(tone) {
        if (tone === 'positive') return 'positive';
        if (tone === 'negative') return 'negative';
        if (tone === 'warning') return 'warning-text';
        return 'neutral';
    }

    function chipToneClass(tone) {
        if (tone === 'positive') return 'chip chip-positive';
        if (tone === 'negative') return 'chip chip-negative';
        if (tone === 'warning') return 'chip chip-warning';
        if (tone === 'off') return 'chip chip-off';
        return 'chip chip-neutral';
    }

    /* riskBarColor comes from lib/risk.js — bound here for legacy call sites */
    var riskBarColor = _RISK.riskBarColor;

    function updateRiskBar(barId, pct) {
        var el = document.getElementById(barId);
        if (!el) return;
        var clamped = _RISK.clampPct ? _RISK.clampPct(pct) : Math.min(Math.max(pct, 0), 100);
        el.style.width = clamped + '%';
        el.className = 'risk-bar-fill ' + riskBarColor(clamped);
    }

    function _formatCooldownEta(sec) {
        sec = Math.max(0, Math.floor(sec || 0));
        if (sec < 60) return sec + 'с';
        var h = Math.floor(sec / 3600);
        var m = Math.floor((sec % 3600) / 60);
        return h > 0 ? (h + 'ч ' + m + 'м') : (m + 'м');
    }

    function renderBlockedStrategies(blocked) {
        var panel = document.getElementById('blocked-strategies-panel');
        var list = document.getElementById('blocked-strategies-list');
        if (!panel || !list) return;
        var names = Object.keys(blocked || {});
        if (names.length === 0) {
            panel.hidden = true;
            list.innerHTML = '';
            return;
        }
        panel.hidden = false;
        list.innerHTML = names.map(function(name) {
            var info = blocked[name] || {};
            var rem = info.remaining_sec || 0;
            var total = info.total_sec || 1;
            var pct = Math.min(100, Math.max(0, (1 - rem / total) * 100));
            return '<div class="kv-row" data-strategy="' + escapeHtml(name) + '">' +
                '<span class="kv-key">' + escapeHtml(name) + '</span>' +
                '<span class="kv-value" style="color:var(--amber)">⏸ ' + _formatCooldownEta(rem) + '</span>' +
                '<div class="risk-bar-track" style="margin-top:6px;width:100%">' +
                    '<div class="risk-bar-fill risk-bar-fill--amber" style="width:' + pct.toFixed(1) + '%"></div>' +
                '</div>' +
                '</div>';
        }).join('');
    }

    /* ── Render functions ────────────────────── */
    function renderKeyValueList(containerId, items) {
        var container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = (items || []).map(function(item) {
            return '<div class="kv-row">' +
                '<span class="kv-key">' + escapeHtml(item.label || '-') + '</span>' +
                '<span class="kv-value ' + toneClass(item.tone) + '">' + escapeHtml(String(item.value || '-')) + '</span>' +
                '</div>';
        }).join('');
    }

    /* ── Toast ───────────────────────────────── */
    var toastTimer = null;
    function showToast(message, type) {
        type = type || 'success';
        var t = document.getElementById('toast');
        t.textContent = message;
        t.className = 'toast toast-' + type + ' show';
        clearTimeout(toastTimer);
        toastTimer = setTimeout(function() { t.className = 'toast'; }, 3500);
    }

    /* ── Market Chart (TradingView Lightweight Charts) ───────────────────
       Pro-grade candlesticks + volume subpane + EMA overlays + SL/TP price
       lines + live updates + historical navigation. Runs only on index.html;
       bails out silently on other pages (container div absent). */
    var chartSeriesMode = 'market';
    var activeInterval = '1h';
    var chartType = 'candle';     /* 'candle' or 'line' */
    var _chartEndTs = 0;          /* 0 = live tail; >0 = historical window end (ms) */
    var activeSymbol = 'BTCUSDT';
    var _indicatorsPerSymbol = {};
    var _tradingSymbols = [];
    var _posTradeFilter = '';
    var _openPositions = [];
    var _userHasZoomed = false;   /* set on user pan/zoom, cleared on reset */

    /* Symbol color mapping for visual distinction */
    var _symbolColors = {
        BTC: { dot: '#f7931a', bg: 'rgba(247,147,26,0.15)', border: 'rgba(247,147,26,0.3)', text: '#f7931a', label: 'BTC' },
        ETH: { dot: '#627eea', bg: 'rgba(98,126,234,0.15)', border: 'rgba(98,126,234,0.3)', text: '#8b9fef', label: 'ETH' },
        SOL: { dot: '#9945ff', bg: 'rgba(153,69,255,0.15)', border: 'rgba(153,69,255,0.3)', text: '#b77dff', label: 'SOL' },
        BNB: { dot: '#f3ba2f', bg: 'rgba(243,186,47,0.15)', border: 'rgba(243,186,47,0.3)', text: '#f3ba2f', label: 'BNB' },
        XRP: { dot: '#00aae4', bg: 'rgba(0,170,228,0.15)', border: 'rgba(0,170,228,0.3)', text: '#4dc4ed', label: 'XRP' },
    };
    var _defaultSymColor = { dot: '#8892a8', bg: 'rgba(136,146,168,0.12)', border: 'rgba(136,146,168,0.25)', text: '#b8c0d4', label: '?' };

    function _getSymColor(symbol) {
        var short = (symbol || '').replace('USDT', '').replace('BUSD', '');
        return _symbolColors[short] || _defaultSymColor;
    }

    function _symBadgeHtml(symbol) {
        var c = _getSymColor(symbol);
        var short = (symbol || '').replace('USDT', '');
        return '<span class="sym-badge" style="background:' + c.bg + ';border:1px solid ' + c.border + ';color:' + c.text + ';">' +
            '<span class="sym-badge-dot" style="background:' + c.dot + ';"></span>' + escapeHtml(short) + '</span>';
    }

    /* LWC chart + series handles (null on non-dashboard pages) */
    var lwcChart = null;
    var candleSeries = null;
    var lineSeries = null;    /* activated when chartType='line' */
    var volumeSeries = null;
    var ema9Series = null;
    var ema21Series = null;
    var _positionPriceLines = []; /* handles returned by createPriceLine for SL/TP/entry */
    var _lastCandles = [];    /* full candle set from backend (for tooltip + price hero) */

    (function initLwcChart() {
        var container = document.getElementById('lwcChart');
        if (!container || typeof LightweightCharts === 'undefined') return;

        lwcChart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: container.clientHeight || 460,
            layout: {
                background: { type: 'solid', color: 'transparent' },
                textColor: '#8892a8',
                fontFamily: 'JetBrains Mono, ui-monospace, monospace',
                fontSize: 10,
            },
            grid: {
                vertLines: { color: 'rgba(56, 68, 91, 0.18)' },
                horzLines: { color: 'rgba(56, 68, 91, 0.18)' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: 'rgba(140, 149, 168, 0.55)', style: LightweightCharts.LineStyle.Dashed, width: 1, labelBackgroundColor: '#6366f1' },
                horzLine: { color: 'rgba(140, 149, 168, 0.55)', style: LightweightCharts.LineStyle.Dashed, width: 1, labelBackgroundColor: '#6366f1' },
            },
            rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.08, bottom: 0.28 } },
            timeScale: {
                borderVisible: false,
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 6,
                barSpacing: 8,
            },
            handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
            handleScale:  { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
            localization: {
                priceFormatter: function(p) { return formatUsd(p); },
            },
        });

        candleSeries = lwcChart.addCandlestickSeries({
            upColor: '#22c55e', downColor: '#ef4444',
            borderUpColor: '#22c55e', borderDownColor: '#ef4444',
            wickUpColor: '#22c55e',   wickDownColor: '#ef4444',
            priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
        });

        /* Volume histogram on a separate pane at the bottom (20% of height) */
        volumeSeries = lwcChart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
            color: 'rgba(99, 102, 241, 0.45)',
        });
        lwcChart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.78, bottom: 0 },
            borderVisible: false,
        });

        /* EMA overlays on the price pane */
        ema9Series = lwcChart.addLineSeries({
            color: 'rgba(226, 167, 47, 0.85)',
            lineWidth: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: true,
        });
        ema21Series = lwcChart.addLineSeries({
            color: 'rgba(192, 132, 252, 0.75)',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: true,
        });

        /* Track user zoom/pan — disables snap-to-latest on polling refresh */
        lwcChart.timeScale().subscribeVisibleTimeRangeChange(function() {
            /* This fires on every refresh too, so we differentiate via a flag
               that's only set when the mouse/wheel drove the change. */
        });
        container.addEventListener('wheel', function() { _userHasZoomed = true; }, { passive: true });
        container.addEventListener('mousedown', function(e) {
            if (e.button === 0) _userHasZoomed = true;
        });
        container.addEventListener('touchstart', function() { _userHasZoomed = true; }, { passive: true });

        /* Crosshair tooltip — shows O/H/L/C/V + EMA at hover */
        var tooltipEl = document.getElementById('lwcTooltip');
        lwcChart.subscribeCrosshairMove(function(param) {
            if (!tooltipEl) return;
            if (!param || !param.time || !param.point || param.point.x < 0 || param.point.y < 0) {
                tooltipEl.style.display = 'none';
                return;
            }
            var bar = param.seriesData.get(candleSeries);
            if (!bar) { tooltipEl.style.display = 'none'; return; }
            var ema9 = param.seriesData.get(ema9Series);
            var ema21 = param.seriesData.get(ema21Series);
            var vol = param.seriesData.get(volumeSeries);
            var tsSec = typeof param.time === 'object' ? param.time.timestamp : param.time;
            var dt = new Date(tsSec * 1000);
            var pad = function(n) { return n < 10 ? '0' + n : '' + n; };
            var tsTxt = dt.getFullYear() + '-' + pad(dt.getMonth() + 1) + '-' + pad(dt.getDate()) +
                        ' ' + pad(dt.getHours()) + ':' + pad(dt.getMinutes());
            var html = '<div class="lwc-tooltip-ts">' + escapeHtml(tsTxt) + '</div>' +
                '<div class="lwc-tooltip-row"><span>O</span><span>' + formatUsd(bar.open) + '</span></div>' +
                '<div class="lwc-tooltip-row"><span>H</span><span>' + formatUsd(bar.high) + '</span></div>' +
                '<div class="lwc-tooltip-row"><span>L</span><span>' + formatUsd(bar.low) + '</span></div>' +
                '<div class="lwc-tooltip-row"><span>C</span><span>' + formatUsd(bar.close) + '</span></div>';
            if (vol && vol.value) html += '<div class="lwc-tooltip-row"><span>Vol</span><span>' + formatVol(vol.value) + '</span></div>';
            if (ema9 && ema9.value != null) html += '<div class="lwc-tooltip-row" style="color:#e2a72f"><span>EMA 9</span><span>' + formatUsd(ema9.value) + '</span></div>';
            if (ema21 && ema21.value != null) html += '<div class="lwc-tooltip-row" style="color:#c084fc"><span>EMA 21</span><span>' + formatUsd(ema21.value) + '</span></div>';
            tooltipEl.innerHTML = html;
            tooltipEl.style.display = 'block';
            /* Position tooltip smartly — avoid going off-screen */
            var cw = container.clientWidth;
            var tx = param.point.x + 16;
            if (tx + 170 > cw) tx = param.point.x - 170;
            var ty = param.point.y + 16;
            if (ty + 160 > container.clientHeight) ty = param.point.y - 160;
            tooltipEl.style.left = tx + 'px';
            tooltipEl.style.top  = Math.max(8, ty) + 'px';
        });

        /* Responsive resize — lightweight-charts doesn't auto-resize on container change */
        window.addEventListener('resize', function() {
            if (lwcChart && container) {
                lwcChart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
            }
        });
    })();

    /* ── EMA Legend Toggle (LWC) ── */
    function _setSeriesVisible(series, visible) {
        if (!series) return;
        series.applyOptions({ visible: visible });
    }
    (function initLegendToggles() {
        var l9 = document.getElementById('lwcLegendEma9');
        var l21 = document.getElementById('lwcLegendEma21');
        if (l9)  l9.addEventListener('click',  function() { l9.classList.toggle('dim'); _setSeriesVisible(ema9Series, !l9.classList.contains('dim')); });
        if (l21) l21.addEventListener('click', function() { l21.classList.toggle('dim'); _setSeriesVisible(ema21Series, !l21.classList.contains('dim')); });
    })();

    function _updateLegend(sym, interval, last) {
        var s = document.getElementById('lwcLegendSymbol');
        if (s) s.textContent = (sym || 'BTCUSDT') + ' · ' + (interval || '1h').toUpperCase();
        var v9 = document.getElementById('lwcLegendEma9Val');
        var v21 = document.getElementById('lwcLegendEma21Val');
        if (v9)  v9.textContent  = (last && last.ema9  != null) ? formatUsd(last.ema9)  : '—';
        if (v21) v21.textContent = (last && last.ema21 != null) ? formatUsd(last.ema21) : '—';
    }

    function setChartTitle(text) {
        var el = document.getElementById('pnlChartTitle');
        if (el) el.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>' + escapeHtml(text);
    }

    function hidePriceHero() {
        var header = document.getElementById('chartPriceHeader');
        if (header) header.style.display = 'none';
    }

    function updatePriceHero(candles) {
        var header = document.getElementById('chartPriceHeader');
        if (!header || !candles || !candles.length) { if (header) header.style.display = 'none'; return; }
        var first = candles[0];
        var last = candles[candles.length - 1];
        var openRef = (first && first.o != null) ? first.o : last.o;
        var close = last.c;
        var diff = close - openRef;
        var pct = openRef ? (diff / openRef) * 100 : 0;

        var hi = -Infinity, lo = Infinity, vol = 0;
        for (var i = 0; i < candles.length; i++) {
            var k = candles[i];
            if (k.h != null && k.h > hi) hi = k.h;
            if (k.l != null && k.l < lo) lo = k.l;
            if (k.v != null) vol += k.v;
        }
        if (hi === -Infinity) hi = close;
        if (lo === Infinity) lo = close;
        var rng = hi - lo;
        var rngPct = lo ? (rng / lo) * 100 : 0;

        var priceEl = document.getElementById('chartPriceNow');
        var chEl = document.getElementById('chartPriceChange');
        if (priceEl) priceEl.textContent = formatUsd(close);
        if (chEl) {
            var sign = diff > 0 ? '+' : (diff < 0 ? '' : '');
            var pctSign = pct > 0 ? '+' : '';
            chEl.textContent = sign + formatUsd(diff) + '  ' + pctSign + pct.toFixed(2) + '%';
            chEl.classList.remove('neg', 'flat');
            if (diff < 0) chEl.classList.add('neg');
            else if (diff === 0) chEl.classList.add('flat');
        }

        var hiEl = document.getElementById('chartStatHigh');
        var loEl = document.getElementById('chartStatLow');
        var rngEl = document.getElementById('chartStatRange');
        var volEl = document.getElementById('chartStatVol');
        if (hiEl) hiEl.textContent = formatUsd(hi);
        if (loEl) loEl.textContent = formatUsd(lo);
        if (rngEl) rngEl.textContent = formatUsd(rng) + ' · ' + rngPct.toFixed(2) + '%';
        if (volEl) volEl.textContent = vol > 0 ? formatVol(vol) : '—';

        header.style.display = 'flex';
    }

    /* ── Convert backend candle (t in ms) to LWC bar (time in seconds) ── */
    function _toLwcBar(c) {
        return {
            time: Math.floor(c.t / 1000),
            open: c.o, high: c.h, low: c.l, close: c.c,
        };
    }

    /* ── Render market candles into LWC chart.
       Always operates in "market" mode — PnL fallback was dropped for the
       pro build (PnL history lives in Equity Curve below). */
    async function renderMarketSeries(interval) {
        if (!lwcChart || !candleSeries) return;
        interval = interval || activeInterval;
        var url = '/api/market-chart?interval=' + encodeURIComponent(interval) + '&symbol=' + encodeURIComponent(activeSymbol);
        if (_chartEndTs && _chartEndTs > 0) url += '&end=' + encodeURIComponent(_chartEndTs);

        var market;
        try {
            var r = await fetch(url);
            market = await r.json();
        } catch (e) {
            _logError('renderMarketSeries.fetch', e, { url: url });
            return;
        }
        var noDataEl = document.getElementById('pnlNoData');
        var candles = (market && market.candles) || [];

        if (!candles.length || candles.length < 2) {
            if (noDataEl) noDataEl.style.display = 'flex';
            setChartTitle('Market · нет данных');
            hidePriceHero();
            return;
        }
        if (noDataEl) noDataEl.style.display = 'none';
        chartSeriesMode = 'market';
        _lastCandles = candles;

        /* De-duplicate by timestamp (backend sometimes repeats the boundary bar) */
        var seen = {};
        var bars = [];
        var volBars = [];
        var ema9Pts = [];
        var ema21Pts = [];
        for (var i = 0; i < candles.length; i++) {
            var c = candles[i];
            var tSec = Math.floor(c.t / 1000);
            if (seen[tSec]) continue;
            seen[tSec] = 1;
            bars.push({ time: tSec, open: c.o, high: c.h, low: c.l, close: c.c });
            if (c.v && c.v > 0) {
                volBars.push({
                    time: tSec, value: c.v,
                    color: c.c >= c.o ? 'rgba(34, 197, 94, 0.45)' : 'rgba(239, 68, 68, 0.45)',
                });
            }
            if (c.ema9 != null) ema9Pts.push({ time: tSec, value: c.ema9 });
            if (c.ema21 != null) ema21Pts.push({ time: tSec, value: c.ema21 });
        }

        /* Toggle candle vs line mode */
        if (chartType === 'line') {
            if (!lineSeries) {
                lineSeries = lwcChart.addAreaSeries({
                    lineColor: '#22c55e',
                    topColor: 'rgba(34, 197, 94, 0.32)',
                    bottomColor: 'rgba(34, 197, 94, 0.0)',
                    lineWidth: 2,
                });
            }
            candleSeries.applyOptions({ visible: false });
            lineSeries.applyOptions({ visible: true });
            lineSeries.setData(bars.map(function(b) { return { time: b.time, value: b.close }; }));
        } else {
            candleSeries.applyOptions({ visible: true });
            if (lineSeries) lineSeries.applyOptions({ visible: false });
            candleSeries.setData(bars);
        }

        volumeSeries.setData(volBars);
        ema9Series.setData(ema9Pts);
        ema21Series.setData(ema21Pts);

        /* Fit view only if user hasn't manually zoomed/panned */
        if (!_userHasZoomed) {
            try { lwcChart.timeScale().fitContent(); } catch(_e) {}
        }

        setChartTitle((market.symbol || activeSymbol) + ' · ' + (market.interval || interval).toUpperCase());
        updatePriceHero(candles);
        _updateLegend(market.symbol || activeSymbol, market.interval || interval, candles[candles.length - 1]);

        /* Re-apply position overlays (price lines move when candles reload) */
        _applyPositionOverlays();
    }

    /* ── Interval Buttons ───────────────────── */
    (function initIntervalBar() {
        var bar = document.getElementById('intervalBar');
        if (!bar) return;
        bar.addEventListener('click', function(e) {
            var btn = e.target.closest('.interval-btn');
            if (!btn) return;
            var interval = btn.getAttribute('data-interval');
            if (!interval || interval === activeInterval) return;

            /* Update active state */
            bar.querySelectorAll('.interval-btn').forEach(function(b) {
                b.classList.remove('active');
                b.setAttribute('aria-checked', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-checked', 'true');
            activeInterval = interval;
            _userHasZoomed = false;
            if (lwcChart) { try { lwcChart.timeScale().fitContent(); } catch(_e) {} }

            /* Fetch new data */
            renderMarketSeries(interval);
        });
    })();

    /* ── Symbol Selector Buttons ────────────── */
    (function initSymbolBar() {
        var bar = document.getElementById('symbolBar');
        if (!bar) return;
        bar.addEventListener('click', function(e) {
            var btn = e.target.closest('.symbol-btn');
            if (!btn) return;
            var sym = btn.getAttribute('data-symbol');
            if (!sym || sym === activeSymbol) return;

            bar.querySelectorAll('.symbol-btn').forEach(function(b) {
                b.classList.remove('active');
                b.setAttribute('aria-checked', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-checked', 'true');
            activeSymbol = sym;
            _userHasZoomed = false;
            if (lwcChart) { try { lwcChart.timeScale().fitContent(); } catch(_e) {} }

            /* Re-fetch chart for new symbol */
            renderMarketSeries(activeInterval);
            /* Update indicators immediately from cached data */
            var symInd = _indicatorsPerSymbol[activeSymbol];
            if (symInd && symInd.symbol) {
                var symWr = (typeof _winRatePerSymbol[activeSymbol] === 'number') ? _winRatePerSymbol[activeSymbol] : (_lastWinRate || 0);
                updateIndicators(symInd, symWr, _lastTradesToday || 0);
            }
            /* Update positions & trades symbol filter to match */
            _posTradeFilter = sym;
            _updateSymFilterBars();
            fetchPositions();
            fetchTrades();
        });
    })();

    /* Build symbol bar dynamically from backend config */
    function _rebuildSymbolBar(symbols) {
        if (!symbols || !symbols.length) return;
        if (JSON.stringify(symbols) === JSON.stringify(_tradingSymbols)) return;
        _tradingSymbols = symbols;

        var bar = document.getElementById('symbolBar');
        if (!bar) return;

        var dotColors = { BTC: 'btc', ETH: 'eth' };
        bar.innerHTML = symbols.map(function(sym) {
            var short = sym.replace('USDT', '');
            var dotClass = dotColors[short] || '';
            var isActive = sym === activeSymbol;
            return '<button class="symbol-btn' + (isActive ? ' active' : '') + '" data-symbol="' + escapeHtml(sym) + '" role="radio" aria-checked="' + isActive + '">' +
                (dotClass ? '<span class="symbol-dot ' + dotClass + '"></span>' : '') +
                escapeHtml(short) + '</button>';
        }).join('');

        // If activeSymbol not in list, select first
        if (symbols.indexOf(activeSymbol) === -1) {
            activeSymbol = symbols[0];
            var firstBtn = bar.querySelector('.symbol-btn');
            if (firstBtn) { firstBtn.classList.add('active'); firstBtn.setAttribute('aria-checked', 'true'); }
        }
    }

    var _lastWinRate = 0;
    var _lastTradesToday = 0;
    var _winRatePerSymbol = {};

    /* ── Symbol Filter Bars for Positions & Trades ── */
    function _buildSymFilterBar(containerId) {
        var container = document.getElementById(containerId);
        if (!container) return;
        var existing = container.querySelector('.sym-filter-bar');
        if (existing) existing.remove();

        if (_tradingSymbols.length < 2) return; /* No filter needed for single symbol */

        var bar = document.createElement('div');
        bar.className = 'sym-filter-bar';

        /* "All" button */
        var allBtn = document.createElement('button');
        allBtn.className = 'sym-filter-btn' + (!_posTradeFilter ? ' active' : '');
        allBtn.textContent = 'Все';
        allBtn.onclick = function() {
            _posTradeFilter = '';
            _updateSymFilterBars();
            fetchPositions();
            fetchTrades();
        };
        bar.appendChild(allBtn);

        /* Per-symbol buttons */
        _tradingSymbols.forEach(function(sym) {
            var c = _getSymColor(sym);
            var short = sym.replace('USDT', '');
            var btn = document.createElement('button');
            btn.className = 'sym-filter-btn' + (_posTradeFilter === sym ? ' active' : '');
            btn.innerHTML = '<span class="sym-filter-dot" style="background:' + c.dot + ';box-shadow:0 0 6px ' + c.dot + ';"></span>' + escapeHtml(short);
            btn.onclick = function() {
                _posTradeFilter = sym;
                _updateSymFilterBars();
                fetchPositions();
                fetchTrades();
            };
            bar.appendChild(btn);
        });

        /* Insert before the first child (after panel-title, but before content) */
        var insertRef = container.querySelector('.positions-header-row') || container.querySelector('.panel-desc') || container.querySelector('table');
        if (insertRef) {
            insertRef.parentNode.insertBefore(bar, insertRef);
        } else {
            container.appendChild(bar);
        }
    }

    function _updateSymFilterBars() {
        _buildSymFilterBar('positions-panel-wrap');
        _buildSymFilterBar('trades-panel-wrap');
    }

    /* ── Chart Type Buttons ──────────────────── */
    (function initChartTypeBar() {
        var bar = document.getElementById('chartTypeBar');
        if (!bar) return;
        bar.addEventListener('click', function(e) {
            var btn = e.target.closest('.chart-type-btn');
            if (!btn) return;
            var type = btn.getAttribute('data-type');
            if (!type || type === chartType) return;

            bar.querySelectorAll('.chart-type-btn').forEach(function(b) {
                b.classList.remove('active');
                b.setAttribute('aria-checked', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-checked', 'true');
            chartType = type;
            _userHasZoomed = false;
            if (lwcChart) { try { lwcChart.timeScale().fitContent(); } catch(_e) {} }

            /* Re-render with new type */
            renderMarketSeries(activeInterval);
        });
    })();

    /* ── UI Updates ──────────────────────────── */
    /* Coalesce rapid state_update ticks — if the WS reconnects while a
       REST fallback fetch is in-flight, both would call updateUI back-to-back
       with essentially identical payloads. 150ms is below human-perceptible.
       `_lastUpdateAt = -Infinity` so the VERY FIRST update always renders,
       even if it arrives <150ms after page load. */
    var _lastUpdateAt = -Infinity;
    function updateUI(data) {
        var now = (typeof performance !== 'undefined' && performance.now)
            ? performance.now() : Date.now();
        if (now - _lastUpdateAt < 150) return;
        _lastUpdateAt = now;
        return _renderUI(data);
    }

    function _renderUI(data) {
        /* Dashboard-only метрики (index.html). На других страницах их нет. */
        var pt = document.getElementById('pnl-today');
        if (pt) {
            /* Снимаем skeleton-состояние metrics-grid после первого данных */
            var mg = document.getElementById('metrics-grid');
            if (mg && mg.dataset.loading === 'true') mg.dataset.loading = 'false';

            pt.textContent = formatPnl(data.pnl_today || 0);
            pt.className = 'metric-value ' + pnlClass(data.pnl_today || 0);

            document.getElementById('balance').textContent = formatUsd(data.balance || 0);
            document.getElementById('open-positions').textContent = data.open_positions || 0;
            document.getElementById('trades-today').textContent = data.trades_today || 0;

            var wr = data.win_rate || 0;
            var wrEl = document.getElementById('win-rate');
            wrEl.textContent = wr.toFixed(1) + '%';
            wrEl.className = 'metric-value ' + (wr >= 50 ? 'positive' : wr > 0 ? 'negative' : 'neutral');
            var twins = data.total_wins || 0, tlosses = data.total_losses || 0;
            if (twins + tlosses > 0) document.getElementById('win-rate-sub').textContent = twins + 'W / ' + tlosses + 'L';

            var dd = data.max_drawdown_pct || 0;
            var ddEl = document.getElementById('max-drawdown');
            ddEl.textContent = dd.toFixed(1) + '%';
            ddEl.className = 'metric-value ' + (dd > 10 ? 'negative' : dd > 5 ? 'neutral' : 'positive');
            var curDD = data.current_drawdown_pct || 0;
            document.getElementById('max-drawdown-sub').textContent = 'Сейчас: ' + curDD.toFixed(1) + '%';

            var pf = data.profit_factor || 0;
            var pfEl = document.getElementById('profit-factor');
            pfEl.textContent = pf.toFixed(2);
            pfEl.className = 'metric-value ' + (pf >= 1.5 ? 'positive' : pf >= 1.0 ? 'neutral' : pf > 0 ? 'negative' : 'neutral');
            document.getElementById('profit-factor-sub').textContent = pf >= 1.5 ? 'Strong system' : pf >= 1.0 ? 'Breakeven+' : pf > 0 ? 'Losing system' : 'No data';

            var rr = data.avg_rr_ratio || 0;
            var rrEl = document.getElementById('avg-rr');
            rrEl.textContent = rr.toFixed(2);
            rrEl.className = 'metric-value ' + (rr >= 2.0 ? 'positive' : rr >= 1.0 ? 'neutral' : rr > 0 ? 'negative' : 'neutral');
            document.getElementById('avg-rr-sub').textContent = rr >= 2.0 ? 'Excellent R:R' : rr >= 1.0 ? 'Acceptable R:R' : 'Improve R:R';

            var peakBal = data.peak_balance || data.balance || 0;
            var peakEl = document.getElementById('equity-peak-badge');
            if (peakEl) peakEl.textContent = 'Пик: $' + peakBal.toFixed(2);
        }

        /* Uptime — общий для всех страниц */
        var upEl = document.getElementById('uptime-label');
        if (upEl && data.uptime) upEl.textContent = data.uptime;

        /* Version — обновляется из бэкенда */
        if (data.version) {
            var verLabel = document.getElementById('version-label');
            var verFooter = document.getElementById('footer-version');
            if (verLabel) verLabel.textContent = 'v' + data.version;
            if (verFooter) verFooter.textContent = 'v' + data.version;
        }

        /* Mode badge — общий */
        var mb = document.getElementById('mode-badge');
        var mode = (data.mode || 'paper').toLowerCase();
        if (mb) {
            mb.innerHTML = '<span class="badge-dot"></span>' + escapeHtml(mode.toUpperCase());
            mb.className = 'badge badge-' + mode;
        }

        /* Risk state badges */
        var rs = (data.risk_state || 'NORMAL').toLowerCase();
        var rsText = (data.risk_state || 'NORMAL').toUpperCase();

        var sb = document.getElementById('state-badge');
        if (sb) {
            sb.innerHTML = '<span class="badge-dot"></span>' + escapeHtml(rsText);
            sb.className = 'badge badge-' + rs;
        }

        var rb = document.getElementById('risk-state-big');
        if (rb) {
            rb.innerHTML = '<span class="badge-dot"></span>' + escapeHtml(rsText);
            rb.className = 'badge badge-' + rs;
        }

        /* Risk details with progress bars — только на dashboard */
        var dlEl = data.risk_details ? document.getElementById('risk-daily-loss') : null;
        if (dlEl) {
            var rd = data.risk_details;
            var lim = rd.limits || {};

            /* Drop skeleton shimmer on first data arrival */
            var rgrid = document.getElementById('risk-grid');
            if (rgrid && rgrid.dataset.loading === 'true') rgrid.dataset.loading = 'false';

            /* Helper: красит значение + надпись-порог по % использования лимита.
               Delegates to lib/risk.js (unit-tested thresholds). */
            var _riskClass = _RISK.riskTextClass || function(p) {
                return p >= 75 ? 'risk-red' : p >= 40 ? 'risk-amber' : 'risk-green';
            };
            function _setThreshold(id, text) {
                var el = document.getElementById(id);
                if (el) el.innerHTML = text;
            }

            /* Daily loss */
            var dailyLossAbs = Math.abs(rd.daily_loss || 0);
            var dailyLimit = lim.max_daily_loss_usd || 0;
            var dailyPct = dailyLimit > 0 ? (dailyLossAbs / dailyLimit) * 100 : 0;
            dlEl.textContent = formatPnl(rd.daily_loss || 0);
            dlEl.className = 'risk-metric-value ' + _riskClass(dailyPct);
            updateRiskBar('risk-daily-loss-bar', Math.min(dailyPct, 100));
            _setThreshold('risk-daily-loss-threshold',
                dailyLimit > 0
                    ? 'Лимит: <b>$' + dailyLimit.toFixed(0) + '</b> · использовано ' + dailyPct.toFixed(0) + '%'
                    : 'Лимит не задан');

            /* Max drawdown */
            var ddPct = ((rd.max_drawdown || 0) * 100);
            var ddEl = document.getElementById('risk-max-dd');
            ddEl.textContent = ddPct.toFixed(1) + '%';
            var ddNorm = (ddPct / 15) * 100; /* 15% считаем "красной" зоной */
            ddEl.className = 'risk-metric-value ' + _riskClass(Math.min(ddNorm, 100));
            updateRiskBar('risk-max-dd-bar', Math.min(ddNorm, 100));

            /* Exposure */
            var expPct = ((rd.exposure || 0) * 100);
            var expLimit = lim.max_total_exposure_pct || 50;
            var expUsedPct = (expPct / expLimit) * 100;
            var expEl = document.getElementById('risk-exposure');
            expEl.textContent = expPct.toFixed(1) + '%';
            expEl.className = 'risk-metric-value ' + _riskClass(expUsedPct);
            updateRiskBar('risk-exposure-bar', Math.min(expUsedPct, 100));
            _setThreshold('risk-exposure-threshold',
                'Лимит: <b>' + expLimit.toFixed(0) + '%</b> · использовано ' + expUsedPct.toFixed(0) + '%');

            /* Trade frequency */
            var freq = rd.trade_freq || 0;
            var freqLimit = lim.max_trades_per_hour || 8;
            var freqPct = (freq / freqLimit) * 100;
            var freqEl = document.getElementById('risk-freq');
            freqEl.textContent = freq + '/ч';
            freqEl.className = 'risk-metric-value ' + _riskClass(freqPct);
            updateRiskBar('risk-freq-bar', Math.min(freqPct, 100));
            _setThreshold('risk-freq-threshold',
                'Лимит: <b>' + freqLimit + '/ч</b>');

            /* Data age — критичный сигнал: торговля по устаревшим ценам = слив */
            var dataAge = rd.market_data_age_sec;
            var dataAgePct = dataAge >= 0 ? Math.min((dataAge / 30) * 100, 100) : 0;
            var dataAgeEl = document.getElementById('risk-data-age');
            dataAgeEl.textContent = dataAge >= 0 ? Number(dataAge).toFixed(1) + 's' : '\u2014';
            dataAgeEl.className = 'risk-metric-value ' + _riskClass(dataAgePct);
            updateRiskBar('risk-data-age-bar', dataAgePct);

            /* Обновляем body[data-stale] + баннер */
            var staleState = '';
            if (dataAge >= 30) staleState = 'critical';
            else if (dataAge >= 10) staleState = 'warn';
            if (staleState) {
                document.body.setAttribute('data-stale', staleState);
                var banner = document.getElementById('data-age-banner-text');
                if (banner) {
                    banner.textContent = staleState === 'critical'
                        ? 'КРИТИЧНО: данные биржи устарели на ' + dataAge.toFixed(0) + 'с — торговля приостановлена'
                        : 'Внимание: задержка данных биржи ' + dataAge.toFixed(0) + 'с (норма <5с)';
                }
            } else {
                document.body.removeAttribute('data-stale');
            }

            /* Commission */
            var commission = rd.daily_commission || 0;
            var totalPnl = Math.abs(data.pnl_today || 0) + commission;
            var commRatio = totalPnl > 0 ? (commission / totalPnl) * 100 : 0;
            var commEl = document.getElementById('risk-commission');
            commEl.textContent = formatUsd(commission);
            commEl.className = 'risk-metric-value ' + _riskClass(commRatio >= 20 ? 80 : commRatio >= 5 ? 50 : 10);
            /* Bar reflects commRatio (% of turnover), not raw $ amount */
            updateRiskBar('risk-commission-bar', Math.min(commRatio * 2, 100));
            _setThreshold('risk-commission-threshold',
                commission > 0
                    ? 'Доля от оборота: <b>' + commRatio.toFixed(1) + '%</b>'
                    : 'Комиссий ещё не было');

            /* Blocked strategies (CB-2 per-strategy isolation) */
            renderBlockedStrategies(rd.blocked_strategies || {});
        }

        /* Footer last update */
        var fuEl = document.getElementById('footer-update');
        if (fuEl) fuEl.textContent = 'Обновлено: ' + new Date().toLocaleTimeString('ru-RU', { hour12: false, timeZone: 'Asia/Dubai' });

        /* Live Activity panel — dashboard-only */
        var wsEl = data.activity ? document.getElementById('act-ws-status') : null;
        if (wsEl) {
            var act = data.activity;
            var col = act.collector || {};
            var prices = col.last_prices || {};

            if (col.connected) {
                wsEl.textContent = 'Онлайн';
                wsEl.className = 'activity-value live';
                document.getElementById('act-ws-sub').textContent = (col.symbols || []).join(', ');
            } else {
                wsEl.textContent = 'Офлайн';
                wsEl.className = 'activity-value';
            }

            // Prices — dynamic per trading symbols (not hardcoded BTC/ETH).
            // Use closest('.activity-item') so label lookup is robust to grid reordering.
            var syms = _tradingSymbols.length ? _tradingSymbols : ['BTCUSDT', 'ETHUSDT'];
            var priceIds = ['act-btc-price', 'act-eth-price'];
            var subIds = ['act-btc-sub', 'act-eth-sub'];
            for (var pi = 0; pi < priceIds.length; pi++) {
                var pel = document.getElementById(priceIds[pi]);
                var sub = document.getElementById(subIds[pi]);
                if (!pel) continue;
                if (pi >= syms.length) {
                    pel.textContent = '—';
                    if (sub) sub.textContent = '';
                    continue;
                }
                var sym = syms[pi];
                var short = sym.replace('USDT', '').replace('BUSD', '');
                var p = prices[sym] || prices[sym.toLowerCase()];
                var item = pel.closest('.activity-item');
                var lblNode = item ? item.querySelector('.activity-label') : null;
                if (lblNode) lblNode.textContent = 'Цена ' + short;
                pel.textContent = p
                    ? '$' + Number(p).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                    : '—';
                if (sub) sub.textContent = p ? (sym + ' · live') : 'ждём данные';
            }

            // Counters
            var msgCount = col.msg_count || 0;
            document.getElementById('act-msg-count').textContent = msgCount.toLocaleString();
            if (col.data_age_sec != null) {
                document.getElementById('act-msg-rate').textContent = 'data age: ' + col.data_age_sec + 's';
            }

            document.getElementById('act-trade-count').textContent = (col.trade_count || 0).toLocaleString();
            document.getElementById('act-candle-count').textContent = (col.candle_count || 0).toLocaleString();
            document.getElementById('act-candle-sub').textContent = (col.candle_closed || 0) + ' closed';

            // Regime
            var regime = act.current_regime;
            var regimeEl = document.getElementById('act-regime');
            if (regime) {
                regimeEl.textContent = regime.replace('_', ' ');
                document.getElementById('act-regime-sub').textContent = 'определён';
            } else {
                regimeEl.textContent = '—';
                document.getElementById('act-regime-sub').textContent = 'ждём данные';
            }

            // Strategies
            var strats = act.strategies_loaded || [];
            document.getElementById('act-strategies').textContent = strats.length;
            document.getElementById('act-strat-sub').textContent = strats.length > 0 ? strats.join(', ') : 'none loaded';

            // Pulse dot
            var dot = document.getElementById('activity-dot');
            if (col.connected && msgCount > 0) {
                dot.style.background = 'var(--green)';
            } else {
                dot.style.background = 'var(--text-muted)';
            }
        }

        /* Start/Stop button state — dashboard-only */
        var paused = data.trading_paused;
        var btnStart = document.getElementById('btn-start');
        var btnStop = document.getElementById('btn-stop');
        if (btnStart && btnStop) {
            if (paused) {
                btnStart.disabled = false;
                btnStart.style.opacity = '1';
                btnStop.disabled = true;
                btnStop.style.opacity = '0.5';
            } else {
                btnStart.disabled = true;
                btnStart.style.opacity = '0.5';
                btnStop.disabled = false;
                btnStop.style.opacity = '1';
            }
        }

        /* ── Per-symbol indicators + symbol bar rebuild ── */
        if (data.indicators_per_symbol) {
            _indicatorsPerSymbol = data.indicators_per_symbol;
        }
        if (data.trading_symbols) {
            _rebuildSymbolBar(data.trading_symbols);
        }
        _lastWinRate = data.win_rate || 0;
        _lastTradesToday = data.trades_today || 0;
        if (data.win_rate_per_symbol) {
            _winRatePerSymbol = data.win_rate_per_symbol;
        }
        var selectedInd = (_indicatorsPerSymbol[activeSymbol] && _indicatorsPerSymbol[activeSymbol].symbol)
            ? _indicatorsPerSymbol[activeSymbol]
            : (data.indicators || {});
        var symWinRate = (typeof _winRatePerSymbol[activeSymbol] === 'number') ? _winRatePerSymbol[activeSymbol] : (data.win_rate || 0);
        updateIndicators(selectedInd, symWinRate, data.trades_today || 0);

        /* ── Readiness Progress panel ── */
        updateReadiness(data.readiness || {});

        /* ── Strategy Decision Log ── */
        updateStrategyLog(data.strategy_log || []);

        /* ── Live Forecast Panel (use per-symbol indicators if available) ── */
        updateForecast(data.strategy_log || [], selectedInd, data);

        /* ── Market Overview (top widget) ── */
        updateMarketOverview(_indicatorsPerSymbol, data.trading_symbols || [], data.strategy_log || [], data.ml_status || {}, data.standing_ml_per_symbol || {}, data.last_cycle_ts_per_symbol || {});

        /* ── Live-update last candle from WS price feed ── */
        try {
            var livePrices = (data.activity && data.activity.collector && data.activity.collector.last_prices) || {};
            var livePrice = livePrices[activeSymbol];
            if (livePrice) _liveUpdateLastCandle(Number(livePrice));
        } catch(e) {
            _logError('liveUpdateLastCandle', e);
        }
    }

    /* ── History navigation ─────────────────────
       When _chartEndTs > 0 the chart is in historical mode:
       - live price ticks are suppressed (_liveUpdateLastCandle bails)
       - the 30s market poll is skipped
       - the history bar shows a yellow HIST badge and a "→ Сейчас" button
       Switching interval/symbol/type does NOT reset history — user is still
       browsing the same window, just at a different zoom. Clicking the live
       button (or reset-zoom) clears _chartEndTs and resumes live polling. */
    function _setHistoryEnd(endMs) {
        _chartEndTs = endMs || 0;
        var badge = document.getElementById('chartHistBadge');
        var liveBtn = document.getElementById('chartHistLive');
        var picker = document.getElementById('chartHistPicker');
        if (_chartEndTs > 0) {
            if (badge) { badge.textContent = 'HIST'; badge.className = 'chart-hist-badge chart-hist-badge-hist'; }
            if (liveBtn) liveBtn.hidden = false;
            if (picker) {
                try {
                    var d = new Date(_chartEndTs);
                    /* datetime-local wants YYYY-MM-DDTHH:MM in *local* time */
                    var pad = function(n) { return n < 10 ? '0' + n : '' + n; };
                    picker.value = d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
                                   'T' + pad(d.getHours()) + ':' + pad(d.getMinutes());
                } catch(_e) { /* ignore */ }
            }
        } else {
            if (badge) { badge.textContent = 'LIVE'; badge.className = 'chart-hist-badge chart-hist-badge-live'; }
            if (liveBtn) liveBtn.hidden = true;
            if (picker) picker.value = '';
        }
        _userHasZoomed = false;
        if (lwcChart) { try { lwcChart.timeScale().fitContent(); } catch(_e) {} }
        renderMarketSeries(activeInterval);
    }

    (function initHistoryBar() {
        var bar = document.getElementById('chartHistoryBar');
        if (!bar) return;

        bar.addEventListener('click', function(e) {
            var btn = e.target.closest('.chart-hist-btn');
            if (!btn) return;
            if (btn.id === 'chartHistLive') { _setHistoryEnd(0); return; }
            var jump = btn.getAttribute('data-hist-jump');
            if (!jump) return;
            /* Anchor: when already in history mode, step further back from
               the current end; otherwise step back from now. */
            var anchor = _chartEndTs > 0 ? _chartEndTs : Date.now();
            var deltaMs = 0;
            if (jump === '-1h')       deltaMs = 60 * 60 * 1000;
            else if (jump === '-1d')  deltaMs = 24 * 60 * 60 * 1000;
            else if (jump === '-1w')  deltaMs = 7 * 24 * 60 * 60 * 1000;
            else if (jump === '-1mo') deltaMs = 30 * 24 * 60 * 60 * 1000;
            if (deltaMs) _setHistoryEnd(anchor - deltaMs);
        });

        var picker = document.getElementById('chartHistPicker');
        if (picker) {
            picker.addEventListener('change', function() {
                if (!picker.value) { _setHistoryEnd(0); return; }
                var ts = Date.parse(picker.value);
                if (!isNaN(ts)) _setHistoryEnd(ts);
            });
        }
    })();

    /* ── Live-update last candle with fresh WS price (flicks between polls).
       Updates close and extends high/low if breached. X-axis labels stay
       the same (labeled by /api/market-chart), so this is visual-only until
       the next 30s poll rebuilds data with a new bar boundary. */
    function _liveUpdateLastCandle(price) {
        if (!candleSeries || chartSeriesMode !== 'market') return;
        if (_chartEndTs > 0) return;    /* frozen historical view — don't corrupt */
        if (!_lastCandles || !_lastCandles.length) return;
        var last = _lastCandles[_lastCandles.length - 1];
        if (!last) return;

        last.c = price;
        if (price > last.h) last.h = price;
        if (price < last.l) last.l = price;

        var bar = {
            time: Math.floor(last.t / 1000),
            open: last.o, high: last.h, low: last.l, close: last.c,
        };
        try {
            if (chartType === 'line' && lineSeries) {
                lineSeries.update({ time: bar.time, value: bar.close });
            } else {
                candleSeries.update(bar);
            }
        } catch (e) {
            /* LWC throws if time < last bar time — can happen across DST or when
               the poll refreshes to a newer bar between ticks. Silent skip. */
        }

        updatePriceHero(_lastCandles);
    }

    /* ── Apply open-position SL/TP/Entry as LWC priceLines.
       Attaches to whichever series is currently visible (candleSeries in
       candle mode, lineSeries in line mode) — a hidden series hides its
       price lines too. Called by renderMarketSeries after setData and by
       fetchPositions. Removes any previous lines first for consistency. */
    function _applyPositionOverlays() {
        var activeSeries = (chartType === 'line' && lineSeries) ? lineSeries : candleSeries;
        if (!activeSeries) return;

        /* Clear stale lines from BOTH series so switching types doesn't leave ghosts */
        for (var i = 0; i < _positionPriceLines.length; i++) {
            var entry = _positionPriceLines[i];
            try { entry.series.removePriceLine(entry.line); } catch(_e) {}
        }
        _positionPriceLines = [];

        if (!_openPositions || !_openPositions.length) return;
        var positions = _openPositions.filter(function(p) { return p.symbol === activeSymbol; });

        positions.forEach(function(pos) {
            var isBuy = pos.side === 'BUY' || pos.side === 'LONG';
            var ep = pos.entry_price || 0;
            var sl = pos.stop_loss_price || 0;
            var tp = pos.take_profit_price || 0;

            function addLine(price, color, title, style) {
                if (!price) return;
                try {
                    var pl = activeSeries.createPriceLine({
                        price: price, color: color, lineWidth: 1,
                        lineStyle: style != null ? style : LightweightCharts.LineStyle.Dashed,
                        axisLabelVisible: true, title: title,
                    });
                    _positionPriceLines.push({ series: activeSeries, line: pl });
                } catch(_e) { /* ignore — price far outside visible range */ }
            }
            addLine(sl, '#ef4444', 'SL', LightweightCharts.LineStyle.Dashed);
            addLine(tp, '#22c55e', 'TP', LightweightCharts.LineStyle.Dashed);
            addLine(ep, '#6366f1', isBuy ? 'BUY' : 'SELL', LightweightCharts.LineStyle.Solid);
        });
    }

    /* ──────────────────────────────────────────────────────────
       Market Overview: куда идёт рынок по каждому символу + почему.
       Scoring logic lives in static/js/lib/market-score.js (unit-tested);
       this file just imports the browser-global binding.
       ────────────────────────────────────────────────────────── */
    var scoreMarketDirection = (typeof SENTINEL !== 'undefined' && SENTINEL.marketScore)
        ? SENTINEL.marketScore.scoreMarketDirection
        : function() { return { score: 50, verdict: 'flat', reasons: [] }; };

    function formatSymbolLabel(sym) {
        return (sym || '').replace(/USDT$/, '').replace(/USD$/, '');
    }

    function formatPriceMo(v) {
        if (!v && v !== 0) return '—';
        var n = Number(v);
        if (!isFinite(n)) return '—';
        if (n >= 1000) return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 2 });
        if (n >= 1) return '$' + n.toFixed(2);
        return '$' + n.toFixed(4);
    }

    function updateMarketOverview(perSym, tradingSymbols, strategyLog, mlStatus, standingMl, lastCycleTs) {
        var gridEl = document.getElementById('market-overview-grid');
        if (!gridEl) return;

        var symbols = (tradingSymbols && tradingSymbols.length) ? tradingSymbols : Object.keys(perSym || {});
        if (!symbols.length) {
            gridEl.innerHTML = '<div class="market-overview-empty">Нет торгуемых символов</div>';
            return;
        }

        // Fallback — latest ML snapshot per symbol from strategy_log
        // (used only if standing ML is not available for the symbol, e.g. model loading).
        var mlBySym = {};
        var log = strategyLog || [];
        for (var li = log.length - 1; li >= 0; li--) {
            var ev = log[li];
            if (!ev || !ev.symbol) continue;
            if (mlBySym[ev.symbol]) continue;
            if (typeof ev.ml_prob !== 'number') continue;
            mlBySym[ev.symbol] = {
                prob: ev.ml_prob,
                decision: ev.ml_decision || '',
                direction: ev.direction || '',
                result: ev.result || '',
                source: 'log'
            };
        }
        var standing = standingMl || {};
        var lastTs = lastCycleTs || {};
        var mlInfo = mlStatus || {};
        var mlEnabled = !!mlInfo.enabled;
        var mlReady = !!mlInfo.is_ready;
        var mlMode = (mlInfo.mode || 'off').toLowerCase();
        var blockThr = Number(mlInfo.block_threshold) || 0.4;
        var perSymMl = mlInfo.per_symbol_models || {};
        var nowMs = Date.now();

        function fmtAge(ms) {
            if (!ms || ms < 0) return '';
            var s = Math.floor(ms / 1000);
            if (s < 2) return 'сейчас';
            if (s < 60) return s + ' с назад';
            var m = Math.floor(s / 60);
            if (m < 60) return m + ' мин назад';
            var h = Math.floor(m / 60);
            return h + ' ч назад';
        }

        var cardsHtml = '';
        var rendered = 0;
        symbols.forEach(function(sym) {
            var ind = (perSym || {})[sym];
            if (!ind || ind.rsi_14 === undefined) return;
            rendered += 1;

            var r = scoreMarketDirection(ind);
            var verdictLabel = r.verdict === 'up' ? 'UP' : (r.verdict === 'down' ? 'DOWN' : 'FLAT');
            var verdictArrow = r.verdict === 'up' ? '▲' : (r.verdict === 'down' ? '▼' : '■');

            // Score bar: center-anchored (50 = middle). Positive fills right, negative fills left.
            var delta = r.score - 50;
            var fillLeft = delta >= 0 ? 50 : (50 + delta);
            var fillWidth = Math.abs(delta);
            var fillColor = r.verdict === 'up' ? 'var(--green)' : (r.verdict === 'down' ? 'var(--red)' : 'var(--amber)');

            var histVal = Number(ind.macd_histogram) || 0;
            var trendLabel = ind.trend === 'bullish' ? 'bull' : (ind.trend === 'bearish' ? 'bear' : 'flat');
            var trendClass = ind.trend === 'bullish' ? 'pos' : (ind.trend === 'bearish' ? 'neg' : '');
            var histClass = histVal > 0 ? 'pos' : (histVal < 0 ? 'neg' : '');

            var reasonsHtml = r.reasons.map(function(reason) {
                return '<div class="mo-reason ' + reason.sign + '"><span class="mo-reason-dot"></span><span>' + reason.text + '</span></div>';
            }).join('');
            if (!reasonsHtml) reasonsHtml = '<div class="mo-reason"><span class="mo-reason-dot"></span><span>Сигналы нейтральны</span></div>';

            // ── ML прогноз для данного символа ──
            // Приоритет: standing ML (всегда свежий, оценивается каждый цикл) → fallback на strategy_log
            var mlHtml = '';
            var mlSymCfg = perSymMl[sym] || {};
            var mlSymReady = !!mlSymCfg.ready || mlReady;
            var standingRec = standing[sym];
            var hasStanding = standingRec && typeof standingRec.prob === 'number';
            var mlRec = hasStanding
                ? {
                    prob: standingRec.prob,
                    decision: standingRec.decision || '',
                    direction: '',
                    result: '',
                    source: 'standing',
                    ts_ms: Number(standingRec.ts_ms) || 0,
                    ref_strategy: standingRec.ref_strategy || ''
                }
                : mlBySym[sym];

            if (!mlEnabled) {
                mlHtml = ''
                    + '<div class="mo-ml off">'
                    +   '<div class="mo-ml-head"><span class="mo-ml-title">ML прогноз</span>'
                    +     '<span class="mo-ml-badge off">выключен</span></div>'
                    +   '<div class="mo-ml-sub">Модель отключена в настройках</div>'
                    + '</div>';
            } else if (!mlSymReady) {
                mlHtml = ''
                    + '<div class="mo-ml loading">'
                    +   '<div class="mo-ml-head"><span class="mo-ml-title">ML прогноз</span>'
                    +     '<span class="mo-ml-badge loading">загрузка…</span></div>'
                    +   '<div class="mo-ml-sub">Модель ещё не готова для ' + formatSymbolLabel(sym) + '</div>'
                    + '</div>';
            } else if (!mlRec) {
                mlHtml = ''
                    + '<div class="mo-ml wait">'
                    +   '<div class="mo-ml-head"><span class="mo-ml-title">ML прогноз</span>'
                    +     '<span class="mo-ml-badge wait">ждём данные</span></div>'
                    +   '<div class="mo-ml-sub">Модель готова — ждём первый тик с индикаторами</div>'
                    + '</div>';
            } else {
                var pct = Math.round(mlRec.prob * 100);
                var isBlocked = mlRec.decision === 'block' || mlRec.result === 'ml_blocked';
                var verdictText, verdictClass;
                if (isBlocked) { verdictText = 'блок'; verdictClass = 'block'; }
                else if (mlRec.prob >= 0.65) { verdictText = 'сильный сигнал'; verdictClass = 'allow-strong'; }
                else if (mlRec.prob >= blockThr) { verdictText = 'разрешено'; verdictClass = 'allow'; }
                else { verdictText = 'слабо'; verdictClass = 'weak'; }
                var barColor = isBlocked ? 'var(--red)' : (mlRec.prob >= 0.65 ? 'var(--green-bright)' : (mlRec.prob >= blockThr ? 'var(--green)' : 'var(--amber)'));
                var dirLabel = mlRec.direction ? (mlRec.direction === 'BUY' ? 'BUY' : (mlRec.direction === 'SELL' ? 'SELL' : '')) : '';
                var modeTag = '';
                if (mlMode === 'shadow') modeTag = '<span class="mo-ml-mode shadow" title="Shadow: модель предсказывает, но не блокирует сделки">shadow</span>';
                else if (mlMode === 'off') modeTag = '<span class="mo-ml-mode off" title="Off: модель только наблюдает, сделки не фильтрует">наблюдение</span>';

                // Freshness: standing records carry ts_ms — compute age and flag stale (>90s)
                var ageHtml = '';
                var staleClass = '';
                if (mlRec.source === 'standing' && mlRec.ts_ms) {
                    var ageMs = Math.max(0, nowMs - mlRec.ts_ms);
                    var isStale = ageMs > 90000;
                    staleClass = isStale ? ' stale' : ' live';
                    var dotTitle = isStale ? 'Оценка устарела (>90 с) — бот, возможно, не обновляет данные' : 'Свежая оценка';
                    ageHtml = '<span class="mo-ml-freshness' + staleClass + '" title="' + dotTitle + '">'
                        + '<span class="mo-ml-dot"></span>' + fmtAge(ageMs) + '</span>';
                } else if (mlRec.source === 'log') {
                    ageHtml = '<span class="mo-ml-freshness log" title="Исторический сигнал из лога стратегий">'
                        + '<span class="mo-ml-dot"></span>из лога</span>';
                }

                var blockPct = Math.round(blockThr * 100);
                var thrLabel, thrTitle;
                if (isBlocked) {
                    thrLabel = 'блок < ' + blockPct + '%';
                    thrTitle = 'Ниже ' + blockPct + '% ML блокирует сделку';
                } else if (mlRec.prob >= 0.65) {
                    thrLabel = 'сильный ≥ 65%';
                    thrTitle = 'Порог сильного сигнала — 65%. Порог блока — ' + blockPct + '%';
                } else {
                    thrLabel = 'порог ' + blockPct + '%';
                    thrTitle = 'Порог блокировки сделки — ' + blockPct + '%';
                }

                mlHtml = ''
                    + '<div class="mo-ml ' + verdictClass + '">'
                    +   '<div class="mo-ml-head">'
                    +     '<span class="mo-ml-title">ML прогноз' + (dirLabel ? ' · ' + dirLabel : '') + '</span>'
                    +     modeTag
                    +     '<span class="mo-ml-pct">' + pct + '%</span>'
                    +   '</div>'
                    +   '<div class="mo-ml-bar"><div class="mo-ml-bar-fill" style="width:' + Math.min(100, Math.max(0, pct)) + '%; background:' + barColor + ';"></div></div>'
                    +   '<div class="mo-ml-sub">'
                    +     '<span class="mo-ml-badge ' + verdictClass + '">' + verdictText + '</span>'
                    +     '<span class="mo-ml-sep" title="' + thrTitle + '">' + thrLabel + '</span>'
                    +     ageHtml
                    +   '</div>'
                    + '</div>';
            }

            // Data freshness per symbol — when was this symbol last processed by the bot
            var cycleTs = Number(lastTs[sym]) || 0;
            var cycleAgeHtml = '';
            if (cycleTs) {
                var cycleAge = Math.max(0, nowMs - cycleTs);
                var cycleStale = cycleAge > 60000;      // >60s: data pipeline likely stalled
                var cycleWarn = cycleAge > 15000;       // >15s: worth surfacing
                var cycleCls = cycleStale ? 'stale' : (cycleWarn ? 'warn' : 'live');
                var cycleTitle = cycleStale ? 'Данные устарели (>60 с): сбой пайплайна?'
                               : cycleWarn ? 'Данные немного запаздывают'
                               : 'Данные свежие';
                cycleAgeHtml = '<span class="mo-freshness ' + cycleCls + '" title="' + cycleTitle + '">'
                    + '<span class="mo-freshness-dot"></span>' + fmtAge(cycleAge) + '</span>';
            }

            cardsHtml += ''
                + '<div class="mo-card ' + r.verdict + '">'
                +   '<div class="mo-card-head">'
                +     '<div class="mo-card-head-main"><span class="mo-symbol">' + formatSymbolLabel(sym) + '</span> <span class="mo-price">' + formatPriceMo(ind.close) + '</span>' + cycleAgeHtml + '</div>'
                +     '<span class="mo-verdict ' + r.verdict + '"><span class="mo-verdict-arrow">' + verdictArrow + '</span>' + verdictLabel + ' ' + r.score + '%</span>'
                +   '</div>'
                +   '<div>'
                +     '<div class="mo-score-label"><span>DOWN</span><span>FLAT</span><span>UP</span></div>'
                +     '<div class="mo-score-bar"><div class="mo-score-fill" style="left:' + fillLeft + '%; width:' + fillWidth + '%; background:' + fillColor + ';"></div></div>'
                +   '</div>'
                +   '<div class="mo-stats">'
                +     '<div class="mo-stat"><span class="mo-stat-label">Тренд</span><span class="mo-stat-value ' + trendClass + '">' + trendLabel + '</span></div>'
                +     '<div class="mo-stat"><span class="mo-stat-label">RSI 14</span><span class="mo-stat-value">' + (Number(ind.rsi_14) || 0).toFixed(1) + '</span></div>'
                +     '<div class="mo-stat"><span class="mo-stat-label">MACD h</span><span class="mo-stat-value ' + histClass + '">' + histVal.toFixed(2) + '</span></div>'
                +     '<div class="mo-stat"><span class="mo-stat-label">ADX</span><span class="mo-stat-value">' + (Number(ind.adx) || 0).toFixed(1) + '</span></div>'
                +   '</div>'
                +   mlHtml
                +   '<div class="mo-reasons">'
                +     '<div class="mo-reasons-title">Почему ' + verdictLabel + '</div>'
                +     reasonsHtml
                +   '</div>'
                + '</div>';
        });

        if (!rendered) {
            gridEl.innerHTML = '<div class="market-overview-empty">Ждём данные индикаторов…</div>';
            return;
        }
        gridEl.innerHTML = cardsHtml;
    }

    function updateForecast(entries, indicators, fullData) {
        var pctEl = document.getElementById('forecast-pct');
        var ringFill = document.getElementById('forecast-ring-fill');
        var badgeEl = document.getElementById('forecast-badge');
        var factorsEl = document.getElementById('forecast-factors');
        var decisionEl = document.getElementById('forecast-decision');
        var scanTimeEl = document.getElementById('forecast-scan-time');
        var regimeEl = document.getElementById('forecast-regime');
        var pipelineEl = document.getElementById('decision-pipeline');
        var symEl = document.getElementById('forecast-symbol');
        var dirEl = document.getElementById('forecast-direction');

        if (!pctEl || !ringFill) return;

        // ── Show active symbol + обновить цвет chip-точки ──
        if (symEl) {
            symEl.textContent = activeSymbol.replace('USDT', '');
            var chip = symEl.closest('.fv2-symbol-chip');
            if (chip) {
                var c = _getSymColor(activeSymbol);
                chip.style.background = c.bg;
                chip.style.borderColor = c.border;
                chip.style.color = c.text;
                var dot = chip.querySelector('.symbol-dot');
                if (dot) dot.style.background = c.dot;
            }
        }

        // ── Determine and show direction ──
        var forecastDir = 'HOLD';
        var latestForDir = null;
        for (var di = entries.length - 1; di >= 0; di--) {
            if (entries[di].symbol === activeSymbol && (entries[di].event === 'scan' || entries[di].event === 'signal')) { latestForDir = entries[di]; break; }
        }
        if (latestForDir && latestForDir.event === 'signal') {
            var sigS = (latestForDir.strategies || []).find(function(s) { return s.result === 'signal'; });
            forecastDir = (sigS && sigS.direction) ? sigS.direction.toUpperCase() : 'BUY';
        } else if (latestForDir && latestForDir.event === 'scan') {
            forecastDir = 'HOLD';
        }
        if (dirEl) {
            dirEl.style.display = 'inline-flex';
            /* SVG-стрелка поворачивается через CSS по классу .buy/.sell/.hold */
            var arrowSvg = '<svg class="fv2-dir-arrow" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>';
            var holdSvg  = '<svg class="fv2-dir-arrow" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="5" y1="12" x2="19" y2="12"/></svg>';
            if (forecastDir === 'BUY')       { dirEl.className = 'forecast-direction-badge buy';  dirEl.innerHTML = arrowSvg + ' BUY (LONG)'; }
            else if (forecastDir === 'SELL') { dirEl.className = 'forecast-direction-badge sell'; dirEl.innerHTML = arrowSvg + ' SELL (SHORT)'; }
            else                              { dirEl.className = 'forecast-direction-badge hold'; dirEl.innerHTML = holdSvg + ' HOLD'; }
        }
        var rd = (fullData || {}).risk_details || {};
        var ml = (fullData || {}).ml_status || {};
        var limits = rd.limits || {};
        var riskChecks = rd.risk_checks || {};

        var latest = null;
        for (var i = entries.length - 1; i >= 0; i--) {
            if ((entries[i].event === 'scan' || entries[i].event === 'signal') && (!entries[i].symbol || entries[i].symbol === activeSymbol)) { latest = entries[i]; break; }
        }

        // ── Build factors ──
        var factors = [];
        var totalScore = 0;
        var maxPossible = 0;
        var ind = indicators || {};
        var hasData = ind.rsi_14 !== undefined;
        var buyC = ind.buy_conditions || {};
        var confB = ind.confidence_breakdown || {};
        var news = ind.news || {};

        // ═══════════════════════════════════════════════════
        // Weighted scoring model.
        //
        // Each factor contributes (scoreNum × weight). scoreNum is in [-1, +1].
        // probability = normalised weighted average, mapped onto [0, 100].
        //
        // Weights reflect real pipeline importance:
        //   × 3.0 — primary trend signal (EMA crossover, ML prediction)
        //   × 2.0 — multi-TF confirmation (trend alignment)
        //   × 1.5 — momentum/volume confirmation (RSI, volume, MACD, news)
        //   × 1.0 — trend strength (ADX)
        //   × 0.5 — volatility hints (BB squeeze)
        //
        // `isBlocker` makes a factor hard-gate: any active blocker forces
        // decision to BLOCKED and probability cap to ≤25%.
        // ═══════════════════════════════════════════════════
        function addFactor(f) {
            factors.push(f);
            var w = f.weight || 1;
            totalScore += (f.scoreNum || 0) * w;
            maxPossible += w;
        }

        var activeBlockers = [];

        if (hasData) {
            // 1. EMA crossover / trend — primary (×3)
            var hasCross = !!ind.has_crossover;
            var crossStrong = !!ind.crossover_strong;
            var trendBull = ind.trend === 'bullish';
            var trendBear = ind.trend === 'bearish';
            var emaScore = hasCross && crossStrong ? 1 : trendBull ? 0.5 : trendBear ? -1 : 0;
            var emaSent = emaScore > 0.3 ? 'bullish' : emaScore < -0.3 ? 'bearish' : 'neutral';
            addFactor({
                name: 'Тренд (EMA 9/21)', icon: hasCross ? '🔀' : trendBull ? '↗' : trendBear ? '↘' : '→',
                sentiment: emaSent, weight: 3, heavy: true,
                score: hasCross ? (crossStrong ? 'CROSS' : 'weak') : (trendBull ? '↑' : trendBear ? '↓' : '~'),
                scoreNum: emaScore,
                detail: hasCross
                    ? (crossStrong ? 'EMA9 пересекла EMA21 снизу вверх — главный сигнал BUY' : 'Кроссовер есть, но слишком слабый (< ATR×0.1)')
                    : trendBull ? 'EMA9 > EMA21 — тренд вверх, но без кроссовера'
                    : trendBear ? 'EMA9 < EMA21 — тренд вниз'
                    : 'Боковик — цены EMA9 ≈ EMA21',
            });

            // 2. RSI — 3 zones consistent (oversold/neutral/overbought) (×1.5)
            var rsi = ind.rsi_14;
            var rsiScore = rsi < 30 ? 1 : rsi > 70 ? -1 : rsi < 45 ? 0.3 : rsi > 60 ? -0.3 : 0;
            var rsiSent = rsiScore > 0.2 ? 'bullish' : rsiScore < -0.2 ? 'bearish' : 'neutral';
            var rsiLabel = rsi < 30 ? 'перепродан (хороший вход)'
                         : rsi > 70 ? 'перекуплен (BUY заблокирован стратегией)'
                         : rsi < 45 ? 'слабо-бычий'
                         : rsi > 60 ? 'слабо-медвежий'
                         : 'нейтральный';
            addFactor({
                name: 'RSI (14)', icon: rsi < 30 ? '↓' : rsi > 70 ? '↑' : '~',
                sentiment: rsiSent, weight: 1.5,
                score: rsi.toFixed(1), scoreNum: rsiScore,
                detail: 'RSI = ' + rsi.toFixed(1) + ' — ' + rsiLabel,
            });

            // 3. ADX trend strength (×1)
            var adx = ind.adx || 0;
            var adxScore = adx >= 25 ? 1 : adx >= 18 ? 0.3 : 0;
            addFactor({
                name: 'Сила тренда (ADX)', icon: adx >= 25 ? '💪' : adx >= 18 ? '🤏' : '😴',
                sentiment: adxScore > 0.5 ? 'bullish' : 'neutral', weight: 1,
                score: adx.toFixed(1), scoreNum: adxScore,
                detail: 'ADX = ' + adx.toFixed(1) + ' — '
                      + (adx >= 25 ? 'тренд подтверждён (+0.05 к confidence)'
                      : adx >= 18 ? 'тренд средний' : 'тренд слабый, позиция уменьшена'),
            });

            // 4. Volume (×1.5) — <1.0 blocks BUY in strategy
            var volR = ind.volume_ratio || 0;
            var volScore = volR >= 1.3 ? 1 : volR >= 1.0 ? 0.5 : volR >= 0.7 ? -0.3 : -1;
            addFactor({
                name: 'Объём торгов', icon: volR >= 1.3 ? '🔊' : volR >= 1.0 ? '🔉' : '🔇',
                sentiment: volScore > 0.2 ? 'bullish' : volScore < -0.2 ? 'bearish' : 'neutral',
                weight: 1.5,
                score: volR.toFixed(2) + '×', scoreNum: volScore,
                detail: 'Объём ' + volR.toFixed(2) + '× от среднего — '
                      + (volR >= 1.0 ? 'достаточный для входа' : 'слишком тихий (стратегия блокирует BUY)'),
            });

            // 5. MACD histogram (×1.5)
            var macdH = ind.macd_histogram || 0;
            var macdScore = Math.max(-1, Math.min(1, macdH / 50));
            var macdBull = macdH > 0;
            addFactor({
                name: 'MACD гистограмма', icon: macdBull ? '🟩' : '🟥',
                sentiment: macdBull ? 'bullish' : 'bearish', weight: 1.5,
                score: (macdH > 0 ? '+' : '') + macdH.toFixed(2), scoreNum: macdScore,
                detail: macdBull ? 'MACD > 0 — бычий импульс (+0.10 к confidence)' : 'MACD < 0 — медвежий импульс',
            });

            // 6. BB squeeze (×0.5)
            var bbBw = ind.bb_bandwidth || 0;
            var bbSq = bbBw < 0.03;
            addFactor({
                name: 'Волатильность (BB)', icon: bbSq ? '🎯' : '📐',
                sentiment: bbSq ? 'bullish' : 'neutral', weight: 0.5,
                score: (bbBw * 100).toFixed(1) + '%', scoreNum: bbSq ? 1 : 0,
                detail: bbSq ? 'BB сжатие — ожидается сильное движение' : 'BB в норме',
            });

            // 7. Trend alignment across TFs (×2)
            var ta = ind.trend_alignment;
            if (ta !== undefined && ta !== null) {
                var taScore = ta >= 0.8 ? 1 : ta >= 0.6 ? 0.5 : ta <= 0.2 ? -1 : ta <= 0.4 ? -0.3 : 0;
                addFactor({
                    name: 'Совпадение таймфреймов', icon: ta >= 0.8 ? '🎯' : ta <= 0.2 ? '⚠️' : '🔄',
                    sentiment: taScore > 0.2 ? 'bullish' : taScore < -0.2 ? 'bearish' : 'neutral',
                    weight: 2,
                    score: (ta * 100).toFixed(0) + '%', scoreNum: taScore,
                    detail: 'Trend Alignment = ' + (ta*100).toFixed(0) + '% — '
                          + (ta >= 0.8 ? 'все ТФ согласны'
                          : ta >= 0.6 ? 'частичное совпадение'
                          : ta <= 0.2 ? 'ТФ противоречат (риск ложного сигнала)' : 'нейтрально'),
                });
            }

            // 8. News & F&G (×1.5). Critical news = hard blocker.
            if (news.composite_score !== undefined) {
                var ns = news.composite_score;
                var nsCrit = !!news.critical_alert;
                var newsScore = nsCrit ? -1 : Math.max(-1, Math.min(1, ns / 0.5));
                if (nsCrit) activeBlockers.push('Критическая новость');
                addFactor({
                    name: 'Новости / F&G', icon: nsCrit ? '🚨' : ns > 0.1 ? '📰' : ns < -0.1 ? '⚡' : '📋',
                    sentiment: nsCrit || newsScore < -0.2 ? 'bearish' : newsScore > 0.2 ? 'bullish' : 'neutral',
                    weight: 1.5,
                    isBlocker: nsCrit,
                    score: (ns > 0 ? '+' : '') + ns.toFixed(2), scoreNum: newsScore,
                    detail: (nsCrit ? '⛔ Критическое событие блокирует вход. ' : '')
                          + 'Sentiment ' + ns.toFixed(2)
                          + ', F&G ' + (news.fear_greed_index || 50)
                          + (news.dominant_category ? ' · ' + news.dominant_category : ''),
                });
            }
        }

        // 9. ML prediction — the last filter before entering a trade (×3).
        // Backend attaches { ml_prob, ml_decision } to every strategy result
        // so we always see the actual live output, not just blocks.
        var mlLatestEvent = null;
        if (latest && latest.strategies) {
            for (var si = latest.strategies.length - 1; si >= 0; si--) {
                var sS = latest.strategies[si];
                if (sS && typeof sS.ml_prob === 'number') { mlLatestEvent = sS; break; }
                if (sS && sS.result === 'ml_blocked') { mlLatestEvent = sS; break; }
            }
        }
        var mlEnabled = !!ml.enabled;
        var mlReady = !!ml.is_ready;
        var mlMode = (ml.mode || 'off').toLowerCase();
        if (mlEnabled && mlReady && mlMode !== 'off') {
            var mlBlocked = mlLatestEvent && (mlLatestEvent.result === 'ml_blocked' || mlLatestEvent.ml_decision === 'block');
            var mlProb = null;
            if (mlLatestEvent && typeof mlLatestEvent.ml_prob === 'number') {
                mlProb = mlLatestEvent.ml_prob;
            } else if (mlLatestEvent && typeof mlLatestEvent.detail === 'string') {
                var m = /prob=([\d.]+)/.exec(mlLatestEvent.detail);
                if (m) mlProb = parseFloat(m[1]);
            }
            var thr = ml.block_threshold || 0.40;
            var mlScore = mlProb !== null
                ? Math.max(-1, Math.min(1, (mlProb - thr) / Math.max(1 - thr, 0.01)))
                : 0.3;  // mode=block w/o live prediction yet — slight positive
            var mlSent = mlBlocked ? 'bearish' : mlScore > 0.2 ? 'bullish' : mlScore < -0.2 ? 'bearish' : 'neutral';
            if (mlBlocked) activeBlockers.push('ML предиктор');
            addFactor({
                name: 'ML Предиктор' + (mlMode === 'shadow' ? ' (shadow)' : ''),
                icon: mlBlocked ? '🤖' : mlReady ? '✅' : '⏸',
                sentiment: mlSent, weight: 3, heavy: true,
                isBlocker: mlBlocked,
                score: mlProb !== null ? (mlProb * 100).toFixed(0) + '%' : (mlMode === 'shadow' ? 'SHADOW' : 'READY'),
                scoreNum: mlScore,
                detail: mlBlocked
                    ? '⛔ Модель заблокировала вход' + (mlProb !== null ? ' (prob=' + mlProb.toFixed(2) + ')' : '') + '. Порог блокировки ' + thr.toFixed(2) + '.'
                    : mlMode === 'shadow'
                    ? 'Режим shadow — модель только наблюдает, не блокирует.'
                      + (mlProb !== null ? ' Последний prob=' + mlProb.toFixed(2) + '.' : ' Пока нет предсказаний.')
                    : 'Активный фильтр. Порог блокировки ' + thr.toFixed(2) + '.'
                      + (mlProb !== null ? ' Последний prob=' + mlProb.toFixed(2) + '.' : ''),
            });
        } else if (mlEnabled && !mlReady) {
            addFactor({
                name: 'ML Предиктор', icon: '⏳', sentiment: 'neutral', weight: 0.5,
                score: 'LOAD', scoreNum: 0,
                detail: 'Модель ещё не загружена — бот торгует без ML-фильтра.',
            });
        } else if (!mlEnabled) {
            addFactor({
                name: 'ML Предиктор', icon: '⏸', sentiment: 'neutral', weight: 0.5,
                score: 'OFF', scoreNum: 0,
                detail: 'ML отключён в настройках (ANALYZER_ML_ENABLED=false).',
            });
        }

        // 10. Risk Sentinel — hard gates. Any failing check → forced block.
        var rcKeys = [
            ['state_ok',        'Risk state',          'Бот на паузе — kill-switch активен'],
            ['daily_loss_ok',   'Дневной убыток',      'Дневной лимит убытка достигнут — торговля остановлена'],
            ['positions_ok',    'Лимит позиций',       'Открыто максимум позиций — новых не будет'],
            ['exposure_ok',     'Экспозиция',          'Суммарная экспозиция выше лимита'],
            ['daily_trades_ok', 'Дневной счётчик',     'Дневной лимит сделок достигнут'],
            ['hourly_trades_ok','Частота сделок',      'Часовой лимит сделок достигнут'],
            ['cooldown_ok',     'Cooldown',            'Идёт пауза после последней сделки'],
        ];
        var riskFailed = [];
        for (var rkI = 0; rkI < rcKeys.length; rkI++) {
            var rk = rcKeys[rkI];
            if (riskChecks.hasOwnProperty(rk[0]) && riskChecks[rk[0]] === false) riskFailed.push({ key: rk[0], title: rk[1], explain: rk[2] });
        }
        if (riskFailed.length > 0) {
            activeBlockers.push('Risk Sentinel');
            addFactor({
                name: 'Risk Sentinel',
                icon: '🛡', sentiment: 'bearish', weight: 2, isBlocker: true, heavy: true,
                score: riskFailed.length + '×',
                scoreNum: -1,
                detail: '⛔ Заблокировано: ' + riskFailed.map(function(r) { return r.title; }).join(', '),
            });
        } else if (Object.keys(riskChecks).length) {
            addFactor({
                name: 'Risk Sentinel', icon: '🛡', sentiment: 'bullish', weight: 1,
                score: 'OK', scoreNum: 0.5,
                detail: 'Все ' + Object.keys(riskChecks).length + ' проверок пройдены',
            });
        }

        // ── Determine signal/block status from latest strategy_log event ──
        var isSignal = latest && latest.event === 'signal';
        var isScanned = latest && latest.event === 'scan';
        var wasBlocked = false, blockReason = '', signalConfidence = null;
        if (latest && latest.strategies) {
            for (var j = 0; j < latest.strategies.length; j++) {
                var s = latest.strategies[j];
                if (s.result === 'ml_blocked' || s.result === 'rejected') {
                    wasBlocked = true; blockReason = s.detail || ''; break;
                }
                if (s.result === 'signal' && typeof s.confidence === 'number') {
                    signalConfidence = s.confidence;
                }
            }
        }

        // ── Probability ──
        // Weighted: (totalScore + maxPossible) / (2 × maxPossible) → [0, 100].
        // If the strategy actually fired a signal, use its REAL confidence
        // (not Math.random). Blockers cap probability to ≤25.
        var probability = 0;
        if (maxPossible > 0) {
            probability = Math.max(0, Math.min(100, ((totalScore + maxPossible) / (maxPossible * 2)) * 100));
        }
        if (isSignal && signalConfidence !== null) {
            probability = Math.round(signalConfidence * 100);
        }
        if (activeBlockers.length > 0 || wasBlocked) {
            probability = Math.min(probability, 25);
        }
        probability = Math.round(probability);

        // ── Ring gauge ──
        var circumference = 376.99;
        var offset = circumference - (probability / 100) * circumference;
        ringFill.setAttribute('stroke-dashoffset', offset.toFixed(2));
        var ringColor = probability >= 70 ? 'var(--green-bright)' : probability >= 40 ? 'var(--amber)' : 'var(--red)';
        ringFill.setAttribute('stroke', ringColor);
        pctEl.textContent = probability + '%';
        pctEl.style.color = ringColor;

        // ── Badge ──
        if (isSignal) { badgeEl.className = 'forecast-status-badge ready'; badgeEl.textContent = '>> ВХОД ВЫПОЛНЕН'; }
        else if (wasBlocked) { badgeEl.className = 'forecast-status-badge blocked'; badgeEl.textContent = '⛔ ЗАБЛОКИРОВАНО'; }
        else if (probability >= 65) { badgeEl.className = 'forecast-status-badge ready'; badgeEl.textContent = '✅ ГОТОВ К ВХОДУ'; }
        else if (probability >= 35) { badgeEl.className = 'forecast-status-badge wait'; badgeEl.textContent = '⏳ ОЖИДАНИЕ'; }
        else { badgeEl.className = 'forecast-status-badge blocked'; badgeEl.textContent = '🛑 НЕ ВХОДИТЬ'; }

        // ── Factor cards ──
        // Each card: [icon] [name + weight-chip + detail] [score]
        // Blockers get a red-tinted border; high-signal factors get a green one.
        if (factors.length > 0) {
            var fhtml = '';
            for (var k = 0; k < factors.length; k++) {
                var f = factors[k];
                var cardCls = 'forecast-factor';
                if (f.isBlocker) cardCls += ' blocking';
                else if (f.weight >= 2 && f.scoreNum >= 0.7) cardCls += ' strong';
                var weightCls = 'forecast-factor-weight' + (f.isBlocker ? ' gate' : (f.heavy ? ' heavy' : ''));
                var weightLabel = f.isBlocker ? 'GATE' : ('×' + (f.weight || 1));
                fhtml += '<div class="' + cardCls + '">'
                      +   '<div class="forecast-factor-icon ' + f.sentiment + '">' + f.icon + '</div>'
                      +   '<div class="forecast-factor-body">'
                      +     '<div class="forecast-factor-name">'
                      +       escapeHtml(f.name)
                      +       '<span class="' + weightCls + '">' + weightLabel + '</span>'
                      +     '</div>'
                      +     '<div class="forecast-factor-detail">' + escapeHtml(f.detail) + '</div>'
                      +   '</div>'
                      +   '<div class="forecast-factor-score ' + f.sentiment + '">' + escapeHtml(String(f.score)) + '</div>'
                      + '</div>';
            }
            factorsEl.innerHTML = fhtml;
        }

        // ── Decision text ──
        // Заполняем новую структуру .fv2-takeaway (head + body)
        var takeawayHead = 'Решение бота';
        var takeawayBody = '';
        if (isSignal) {
            var sigStrat = (latest.strategies || []).find(function(s) { return s.result === 'signal'; });
            var sigDir = (sigStrat && sigStrat.direction) ? sigStrat.direction.toUpperCase() : 'BUY';
            var dirLabel = sigDir === 'SELL' ? 'SELL (SHORT)' : 'BUY (LONG)';
            var confTxt = signalConfidence !== null
                ? ' · <span class="green">confidence ' + (signalConfidence*100).toFixed(0) + '%</span>'
                : '';
            takeawayHead = 'Сигнал исполнен';
            takeawayBody = '<strong class="green">Бот вошёл в сделку ' + escapeHtml(activeSymbol) + ' ' + dirLabel + '</strong>'
                    + ' через <strong>' + escapeHtml((sigStrat || {}).strategy || '—') + '</strong>' + confTxt
                    + '. Цена: <strong>$' + (latest.price || 0).toLocaleString() + '</strong>.';
        } else if (activeBlockers.length > 0 || wasBlocked) {
            var reasons = activeBlockers.slice();
            if (wasBlocked && blockReason) reasons.push(blockReason);
            takeawayHead = 'Вход заблокирован';
            takeawayBody = '<strong class="red">Блокеры:</strong> '
                    + reasons.map(escapeHtml).join(' · ')
                    + '. Бот ждёт, пока условия улучшатся.';
        } else if (isScanned) {
            var wf = factors.filter(function(f){ return f.scoreNum <= -0.3; });
            var sf = factors.filter(function(f){ return f.scoreNum >= 0.3; });
            takeawayHead = 'HOLD — сигнала нет';
            takeawayBody = 'Проверено <strong>' + (latest.active_strategies||[]).length + '</strong> стратегий на ' + escapeHtml(activeSymbol) + '.';
            if (sf.length) takeawayBody += '<br><span class="green">За вход:</span> ' + sf.map(function(f){return escapeHtml(f.name)}).join(', ') + '.';
            if (wf.length) takeawayBody += '<br><span class="amber">Против:</span> ' + wf.map(function(f){return escapeHtml(f.name)}).join(', ') + '.';
        } else {
            takeawayBody = 'Бот ещё не завершил первый цикл сканирования. Стратегии запускаются при закрытии 1-часовой свечи.';
        }
        decisionEl.innerHTML =
            '<div class="fv2-takeaway-head">' +
                '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>' +
                '<span>' + escapeHtml(takeawayHead) + '</span>' +
            '</div>' +
            '<div class="fv2-takeaway-body">' + takeawayBody + '</div>';

        if (latest && latest.ts) scanTimeEl.textContent = new Date(latest.ts).toLocaleTimeString('ru-RU', { hour12: false, timeZone: 'Asia/Dubai' });
        if (latest && latest.regime) regimeEl.textContent = 'Режим: ' + latest.regime.toUpperCase();

        // ═══════════════════════════════════════════════════
        // DECISION PIPELINE — collapsible accordion
        // ═══════════════════════════════════════════════════
        if (!pipelineEl || !hasData) return;
        var ph = '';
        var chevronSvg = '<svg aria-hidden="true" class="pipeline-chevron" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M7.21 14.77a.75.75 0 0 1 .02-1.06L11.168 10 7.23 6.29a.75.75 0 1 1 1.04-1.08l4.5 4.25a.75.75 0 0 1 0 1.08l-4.5 4.25a.75.75 0 0 1-1.06-.02z" clip-rule="evenodd"/></svg>';
        function chk(ok){return ok?'✅':'❌'} function cls(ok){return ok?'pass':'fail'} function stCls(ok){return ok?'pass':ok===false?'fail':''} function stLbl(ok){return ok?'ПРОЙДЕН':'БЛОК'}
        // Helpers for collapsible sections
        function secOpen(num, title, statusCls, statusText, stepCls) {
            return '<div class="pipeline-section" onclick="var s=this.parentElement.querySelectorAll(\'.pipeline-section.open\');for(var i=0;i<s.length;i++){if(s[i]!==this)s[i].classList.remove(\'open\');}this.classList.toggle(\'open\')">' +
                '<div class="pipeline-section-header">' +
                '<div class="pipeline-step-num ' + (stepCls||'') + '">' + num + '</div>' +
                chevronSvg +
                '<div class="pipeline-section-title">' + title + '</div>' +
                '<div class="pipeline-section-status ' + statusCls + '">' + statusText + '</div>' +
                '</div><div class="pipeline-content"><div class="pipeline-content-inner">';
        }
        function secClose(explainHtml) {
            return (explainHtml ? '<div class="pipeline-explain">' + explainHtml + '</div>' : '') +
                '</div></div></div>';
        }

        // ── STEP 1: Entry conditions ──
        var allOk = !!ind.all_buy_conditions_met;
        ph += secOpen('1', 'Условия входа (EMA Crossover RSI)', allOk?'pass':'fail', stLbl(allOk), stCls(allOk));
        ph += '<div class="pipeline-checks">';
        var cItems = [
            {name:'EMA9 пересекла EMA21 снизу вверх', ok:buyC.ema_crossover, ex:'Разница: '+(ind.ema_diff!=null?ind.ema_diff.toFixed(4):'—')+(ind.prev_ema_diff!=null?', пред: '+ind.prev_ema_diff.toFixed(4):'')},
            {name:'Кроссовер достаточно сильный (>= ATR×0.1)', ok:buyC.crossover_strong, ex:'Порог: '+(ind.min_cross_threshold!=null?ind.min_cross_threshold.toFixed(4):'—')},
            {name:'RSI < 70 (не перекуплен)', ok:buyC.rsi_below_70, ex:'RSI = '+(ind.rsi_14!=null?ind.rsi_14.toFixed(1):'—')},
            {name:'Объём >= 1.0x среднего', ok:buyC.volume_above_1x, ex:'Volume = '+(ind.volume_ratio!=null?ind.volume_ratio.toFixed(2):'—')+'x'},
            {name:'Цена > EMA50 (глобальный тренд)', ok:buyC.price_above_ema50, ex:'$'+(ind.close||0).toLocaleString()+' vs EMA50=$'+(ind.ema_50||0).toLocaleString()},
            {name:'Нет критических новостей', ok:buyC.no_critical_news, ex:'Score='+(news.composite_score||0).toFixed(2)+', critical='+(news.critical_alert?'ДА':'нет')},
            {name:'Confidence >= 0.60', ok:buyC.confidence_above_min, ex:'Confidence = '+(confB.total!=null?confB.total.toFixed(3):'—')},
        ];
        for(var ci=0;ci<cItems.length;ci++){var c=cItems[ci];var cOk=!!c.ok;ph+='<div class="pipeline-check"><div class="pipeline-check-icon">'+chk(cOk)+'</div><div class="pipeline-check-name">'+escapeHtml(c.name)+'<br><small style="color:var(--text-dim)">'+escapeHtml(c.ex)+'</small></div><div class="pipeline-check-value '+cls(cOk)+'">'+(cOk?'ДА':'НЕТ')+'</div></div>';}
        ph += '</div>';
        ph += secClose('<strong>Как работает:</strong> Все 7 условий должны быть ✅ одновременно. Если хоть одно ❌ — ордер НЕ создаётся, бот ждёт следующую 1h свечу.');

        // ── STEP 2: Confidence formula ──
        var confTotal = confB.total || 0;
        var confOk = confTotal >= 0.60;
        ph += secOpen('2', 'Расчёт Confidence (уверенность)', confOk?'pass':'fail', (confTotal*100).toFixed(0)+'% '+(confOk?'>= 60%':'< 60%'), stCls(confOk));
        ph += '<div class="pipeline-formula">';
        ph += '<span class="label">base</span>                    <span class="op">=</span> <span class="val">0.50</span>\n';
        ph += '<span class="label">RSI '+((ind.rsi_14||0).toFixed(1))+'</span>';
        ph += (confB.rsi_bonus>0?' <span class="op">+</span> <span class="pass">'+confB.rsi_bonus.toFixed(2)+'</span>  <span class="label">'+(ind.rsi_14<50?'(< 50)':'(50-60)')+'</span>':' <span class="op">+</span> <span class="fail">0.00</span>  <span class="label">(>= 60)</span>')+'\n';
        ph += '<span class="label">Volume '+((ind.volume_ratio||0).toFixed(2))+'x</span>';
        ph += (confB.volume_bonus>0?' <span class="op">+</span> <span class="pass">'+confB.volume_bonus.toFixed(2)+'</span>  <span class="label">'+(ind.volume_ratio>2?'(> 2x)':'(> 1.5x)')+'</span>':' <span class="op">+</span> <span class="fail">0.00</span>  <span class="label">(< 1.5x)</span>')+'\n';
        ph += '<span class="label">EMA50</span>              '+(confB.ema50_bonus>0?'<span class="op">+</span> <span class="pass">0.10</span>  <span class="label">(цена > EMA50)</span>':'<span class="op">+</span> <span class="fail">0.00</span>  <span class="label">(цена <= EMA50)</span>')+'\n';
        ph += '<span class="label">MACD '+((ind.macd_histogram||0).toFixed(2))+'</span>     '+(confB.macd_bonus>0?'<span class="op">+</span> <span class="pass">0.10</span>  <span class="label">(> 0)</span>':'<span class="op">+</span> <span class="fail">0.00</span>  <span class="label">(< 0)</span>')+'\n';
        ph += '<span class="label">ADX '+((ind.adx||0).toFixed(1))+'</span>        '+(confB.adx_bonus>0?'<span class="op">+</span> <span class="pass">0.05</span>  <span class="label">(> 25)</span>':'<span class="op">+</span> <span class="fail">0.00</span>  <span class="label">(<= 25)</span>')+'\n';
        var taB=confB.trend_align_bonus||0;
        ph += '<span class="label">Trend Align '+((ind.trend_alignment||0)*100).toFixed(0)+'%</span> '+(taB>0?'<span class="op">+</span> <span class="pass">'+taB.toFixed(2)+'</span>':taB<0?'<span class="op">-</span> <span class="fail">'+Math.abs(taB).toFixed(2)+'</span>':'<span class="op">+</span> <span class="fail">0.00</span>')+'\n';
        ph += '<span class="label">──────────────────────────────────</span>\n';
        ph += '<span class="label">ИТОГО</span>               <span class="op">=</span> <span class="result">'+confTotal.toFixed(3)+' ('+(confTotal*100).toFixed(1)+'%)</span>  '+(confOk?'<span class="pass">>= 60%</span>':'<span class="fail">< 60%</span>');
        ph += '</div>';
        ph += secClose('<strong>Как для чайника:</strong> Бот начинает с 50% уверенности. Каждый хороший фактор добавляет очки. Набрал >= 60% — сигнал создаётся. Нет — ждёт лучших условий. Максимум 95%.');

        // ── STEP 3: ML Filter ──
        var mlEnabled = !!ml.enabled, mlReady = !!ml.is_ready, mlMode = ml.mode||'off';
        ph += secOpen('3', 'ML Фильтр (RF + LightGBM + XGBoost)', mlEnabled?(mlReady?'pass':'wait'):'wait', mlEnabled?(mlReady?'АКТИВЕН ('+mlMode+')':'Обучается...'):'ВЫКЛЮЧЕН', mlEnabled&&mlReady?'':'wait');
        ph += '<div class="pipeline-checks">';
        ph += '<div class="pipeline-check"><div class="pipeline-check-icon">'+(mlEnabled?'✅':'⏸️')+'</div><div class="pipeline-check-name">ML модуль включён</div><div class="pipeline-check-value '+(mlEnabled?'pass':'neutral')+'">'+(mlEnabled?'ДА':'НЕТ')+'</div></div>';
        ph += '<div class="pipeline-check"><div class="pipeline-check-icon">'+(mlReady?'✅':'⏳')+'</div><div class="pipeline-check-name">Модель обучена и готова</div><div class="pipeline-check-value '+(mlReady?'pass':'neutral')+'">'+(mlReady?'ДА':'НЕТ')+'</div></div>';
        ph += '<div class="pipeline-check"><div class="pipeline-check-icon">🎚️</div><div class="pipeline-check-name">Режим</div><div class="pipeline-check-value neutral">'+mlMode+'</div></div>';
        ph += '<div class="pipeline-check"><div class="pipeline-check-icon">🚫</div><div class="pipeline-check-name">Порог блокировки</div><div class="pipeline-check-value neutral">'+((ml.block_threshold||0.55)*100).toFixed(0)+'%</div></div>';
        ph += '<div class="pipeline-check"><div class="pipeline-check-icon">v</div><div class="pipeline-check-name">Порог уменьшения размера</div><div class="pipeline-check-value neutral">'+((ml.reduce_threshold||0.65)*100).toFixed(0)+'%</div></div>';
        ph += '</div><div class="pipeline-formula">';
        ph += '<span class="label">ML решение:</span>\n';
        ph += '  prob <span class="op"><</span> <span class="val">'+((ml.block_threshold||0.55)*100).toFixed(0)+'%</span>   <span class="op">→</span> <span class="fail">БЛОК (ордер отменён)</span>\n';
        ph += '  <span class="val">'+((ml.block_threshold||0.55)*100).toFixed(0)+'%</span> <span class="op">≤</span> prob <span class="op"><</span> <span class="val">'+((ml.reduce_threshold||0.65)*100).toFixed(0)+'%</span> <span class="op">→</span> <span class="result">REDUCE (размер уменьшен)</span>\n';
        ph += '  prob <span class="op">≥</span> <span class="val">'+((ml.reduce_threshold||0.65)*100).toFixed(0)+'%</span>  <span class="op">→</span> <span class="pass">ALLOW (пропускаем)</span>';
        ph += '</div>';
        ph += secClose('<strong>Как для чайника:</strong> ML смотрит 31 параметр и предсказывает шанс прибыли. Если < '+((ml.block_threshold||0.55)*100).toFixed(0)+'% — блокирует. Режим "shadow" = только логирует, не блокирует.');

        // ── STEP 4: Risk Sentinel ──
        var allRiskOk = riskChecks.state_ok!==false && riskChecks.daily_loss_ok!==false && riskChecks.positions_ok!==false && riskChecks.exposure_ok!==false && riskChecks.daily_trades_ok!==false && riskChecks.hourly_trades_ok!==false && riskChecks.cooldown_ok!==false;
        ph += secOpen('4', 'Risk Sentinel (7 проверок безопасности)', allRiskOk?'pass':'fail', allRiskOk?'ВСЕ ОК':'ЕСТЬ БЛОКИ', stCls(allRiskOk));
        ph += '<div class="pipeline-checks">';
        var rItems = [
            {name:'Risk State != STOP', ok:riskChecks.state_ok, ex:'Состояние: '+((fullData||{}).risk_state||'NORMAL')},
            {name:'Дневной убыток < $'+(limits.max_daily_loss_usd||50).toFixed(0), ok:riskChecks.daily_loss_ok, ex:'Текущий: $'+Math.abs(rd.daily_loss||0).toFixed(2)},
            {name:'Позиций < '+(limits.max_open_positions||2), ok:riskChecks.positions_ok, ex:'Открыто: '+((fullData||{}).open_positions||0)},
            {name:'Экспозиция < '+(limits.max_total_exposure_pct||60)+'%', ok:riskChecks.exposure_ok, ex:'Текущая: '+((rd.exposure||0)*100).toFixed(1)+'%'},
            {name:'Сделок/день < '+(limits.max_daily_trades||6), ok:riskChecks.daily_trades_ok, ex:'Сегодня: '+(rd.daily_trades||(fullData||{}).trades_today||0)},
            {name:'Сделок/час < '+(limits.max_trades_per_hour||2), ok:riskChecks.hourly_trades_ok, ex:'За час: '+(rd.trade_freq||0)},
            {name:'Cooldown (мин. '+((limits.min_trade_interval_sec||1800)/60).toFixed(0)+' мин)', ok:riskChecks.cooldown_ok, ex:(rd.cooldown_remaining_sec>0?'Ждём '+rd.cooldown_remaining_sec+'с':'Готов')},
        ];
        for(var ri=0;ri<rItems.length;ri++){var r=rItems[ri];var rOk=r.ok!==false;ph+='<div class="pipeline-check"><div class="pipeline-check-icon">'+chk(rOk)+'</div><div class="pipeline-check-name">'+escapeHtml(r.name)+'<br><small style="color:var(--text-dim)">'+escapeHtml(r.ex)+'</small></div><div class="pipeline-check-value '+cls(rOk)+'">'+(rOk?'ОК':'БЛОК')+'</div></div>';}
        ph += '</div>';
        ph += secClose('<strong>Как для чайника:</strong> Даже если стратегия хочет купить — Risk Sentinel может запретить. Проверяет: не потеряли слишком много, не слишком часто торгуем, не слишком много денег в рынке. SELL никогда не блокируется.');

        // ── STEP 5: Position Sizing ──
        var atr=ind.atr||0, price=ind.close||1, adxV=ind.adx||0;
        var atrPct=price>0?(atr/price*100):0;
        var volFactor=atrPct>0?Math.max(0.3,Math.min(1.5/atrPct,1.5)):1.0;
        var regDamp=adxV>=30?1.0:(adxV>=20?(0.6+(adxV-20)/10*0.4):0.5);
        var balance=(fullData||{}).balance||500;
        var estBudgetUsd=Math.min(balance*Math.max(2,Math.min(8*volFactor*regDamp,20))/100, limits.max_order_usd||100);

        ph += secOpen('5', 'Размер позиции (Kelly + ATR + Regime)', 'pass', '~$'+estBudgetUsd.toFixed(0), '');
        ph += '<div class="pipeline-formula">';
        ph += '<span class="label">1. Kelly (половинный):</span>\n';
        ph += '   f <span class="op">=</span> (win_rate<span class="op">×</span>b <span class="op">-</span> q) <span class="op">/</span> b <span class="op">×</span> 0.5\n\n';
        ph += '<span class="label">2. Волатильность:</span>\n';
        ph += '   ATR% <span class="op">=</span> <span class="val">'+atr.toFixed(2)+'</span> / <span class="val">'+price.toFixed(2)+'</span> × 100 <span class="op">=</span> <span class="result">'+atrPct.toFixed(2)+'%</span>\n';
        ph += '   vol_factor <span class="op">=</span> 1.5 / '+atrPct.toFixed(2)+' <span class="op">=</span> <span class="result">'+volFactor.toFixed(2)+'</span> <span class="label">(0.3-1.5)</span>\n\n';
        ph += '<span class="label">3. Режим (ADX='+adxV.toFixed(1)+'):</span>\n';
        ph += '   dampener <span class="op">=</span> <span class="result">'+regDamp.toFixed(2)+'</span> <span class="label">('+(adxV>=30?'≥30=100%':adxV>=20?'20-30=60-100%':'<20=50%')+')</span>\n\n';
        ph += '<span class="label">4. Итог:</span> budget$ <span class="op">=</span> min(баланс × kelly × vol × damp, <span class="val">$'+(limits.max_order_usd||100).toFixed(0)+'</span>)\n';
        ph += '   <span class="label">Лимиты: $'+(limits.min_order_usd||10).toFixed(0)+' — $'+(limits.max_order_usd||100).toFixed(0)+', позиция 2-20%</span>';
        ph += '</div>';
        ph += secClose('<strong>Как для чайника:</strong> Бот не ставит всё на одну сделку. Оптимальная ставка считается по Kelly. Волатильный рынок — ставка меньше. Слабый тренд — вдвое меньше. Максимум $'+(limits.max_order_usd||100).toFixed(0)+'.');

        // ── STEP 6: SL/TP ──
        var slPct=atrPct>0?Math.max(1,Math.min(atrPct*2,2.9)):2.5;
        var tpPct=Math.max(1.5,Math.min(slPct*2,15));
        var slPrice=price*(1-slPct/100), tpPrice=price*(1+tpPct/100);
        ph += secOpen('6', 'Stop-Loss / Take-Profit (ATR)', 'pass', 'SL '+slPct.toFixed(1)+'% / TP '+tpPct.toFixed(1)+'%', '');
        ph += '<div class="pipeline-formula">';
        ph += '<span class="label">SL%</span> <span class="op">=</span> ATR% × 2.0 <span class="op">=</span> <span class="val">'+(atrPct*2).toFixed(2)+'%</span> <span class="op">→</span> clamp(1%, 2.9%) <span class="op">=</span> <span class="fail">'+slPct.toFixed(2)+'%</span>\n';
        ph += '<span class="label">TP%</span> <span class="op">=</span> SL% × 2.0 (R:R 1:2) <span class="op">=</span> <span class="pass">'+tpPct.toFixed(2)+'%</span>\n\n';
        ph += '<span class="label">При цене $'+price.toLocaleString()+':</span>\n';
        ph += '  SL <span class="op">=</span> <span class="fail">$'+slPrice.toFixed(2)+'</span>  <span class="label">(макс. убыток)</span>\n';
        ph += '  TP <span class="op">=</span> <span class="pass">$'+tpPrice.toFixed(2)+'</span>  <span class="label">(целевая прибыль)</span>';
        ph += '</div>';
        ph += secClose('<strong>Как для чайника:</strong> SL = максимальный убыток, при котором бот автоматически продаёт. TP = цель прибыли. R:R 1:2 = рискуем $1 чтобы заработать $2. SL ограничен 2.9% — иначе Risk Sentinel отклонит.');

        // Preserve which section is open across re-renders
        var openIdx = -1;
        var oldSections = pipelineEl.querySelectorAll('.pipeline-section');
        for (var oi = 0; oi < oldSections.length; oi++) {
            if (oldSections[oi].classList.contains('open')) { openIdx = oi; break; }
        }
        pipelineEl.innerHTML = ph;
        if (openIdx >= 0) {
            var newSections = pipelineEl.querySelectorAll('.pipeline-section');
            if (newSections[openIdx]) newSections[openIdx].classList.add('open');
        }
    }

    function updateStrategyLog(entries) {
        var listEl = document.getElementById('strat-log-list');
        if (!listEl || !entries || entries.length === 0) {
            return;
        }

        var html = '';
        // newest first
        for (var i = entries.length - 1; i >= 0; i--) {
            var e = entries[i];
            var evtClass = 'log-' + (e.event || 'scan');
            var timeStr = e.ts ? new Date(e.ts).toLocaleTimeString('ru-RU', { hour12: false, timeZone: 'Asia/Dubai' }) : '';

            html += '<div class="strat-log-entry ' + evtClass + '">';
            html += '  <div class="strat-log-header">';
            html += '    <div style="display:flex;align-items:center;gap:8px;">';
            html += '      <span class="strat-log-symbol">' + escapeHtml(e.symbol || '') + '</span>';
            if (e.price) {
                html += '      <span class="strat-log-price">$' + Number(e.price).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2}) + '</span>';
            }
            if (e.regime) {
                html += '      <span class="strat-log-regime">' + escapeHtml(e.regime) + '</span>';
            }
            html += '    </div>';
            html += '    <div class="strat-log-meta">';
            html += '      <span class="strat-log-time">' + escapeHtml(timeStr) + '</span>';
            html += '    </div>';
            html += '  </div>';

            // Message
            html += '  <div class="strat-log-msg">' + escapeHtml(e.msg || '') + '</div>';

            // Strategy tags
            if (e.strategies && e.strategies.length > 0) {
                html += '  <div class="strat-log-strategies">';
                for (var j = 0; j < e.strategies.length; j++) {
                    var s = e.strategies[j];
                    var tagClass = 'strat-tag strat-tag-' + (s.result || 'no_signal');
                    var label = escapeHtml(s.strategy || '');
                    if (s.detail) label += ': ' + escapeHtml(s.detail);
                    html += '<span class="' + tagClass + '">' + label + '</span>';
                }
                html += '  </div>';
            }

            html += '</div>';
        }
        listEl.innerHTML = html;
    }

    function updateReadiness(r) {
        var panel = document.getElementById('readiness-panel');
        var pctEl = document.getElementById('readiness-pct');
        var barEl = document.getElementById('readiness-bar');
        var stepsEl = document.getElementById('readiness-steps');
        var infoEl = document.getElementById('readiness-info-text');

        // Панель готовности есть не на всех страницах
        if (!pctEl || !barEl || !stepsEl || !infoEl) return;

        if (!r || !r.steps || r.steps.length === 0) {
            pctEl.textContent = '—';
            barEl.style.width = '0%';
            stepsEl.innerHTML = '';
            infoEl.textContent = 'Waiting for backend data...';
            return;
        }

        var pct = r.pct || 0;
        var ready = r.ready || false;

        // Main percentage
        pctEl.textContent = ready ? 'READY' : pct.toFixed(0) + '%';
        pctEl.className = 'readiness-pct-big ' + (ready ? 'done' : 'progress');

        // Main bar
        barEl.style.width = pct + '%';
        barEl.className = 'readiness-bar-fill' + (ready ? ' done' : '');

        // Panel glow when ready
        panel.className = 'readiness-panel fade-in' + (ready ? ' ready' : '');

        // Steps
        var html = '';
        for (var i = 0; i < r.steps.length; i++) {
            var s = r.steps[i];
            var done = s.done;
            var hasPct = s.pct != null;
            var stepPct = hasPct ? s.pct : (done ? 100 : 0);
            html += '<div class="readiness-step' + (done ? ' step-done' : '') + '">';
            html += '  <div class="readiness-step-header">';
            html += '    <span class="readiness-step-name">';
            html += '      <span class="step-check ' + (done ? 'done' : 'pending') + '">' + (done ? '✓' : '') + '</span>';
            html += '      ' + escapeHtml(s.name);
            html += '    </span>';
            html += '    <span class="readiness-step-detail">' + escapeHtml(s.detail || '') + '</span>';
            html += '  </div>';
            if (hasPct) {
                html += '  <div class="readiness-step-bar-wrap">';
                html += '    <div class="readiness-step-bar' + (done ? ' bar-done' : '') + '" style="width:' + stepPct + '%"></div>';
                html += '  </div>';
            }
            html += '</div>';
        }
        stepsEl.innerHTML = html;

        // Info text
        if (ready) {
            infoEl.textContent = 'System is ready to trade — all data collected for ' + (r.symbol || '') + ' (' + (r.timeframe || '') + ')';
        } else {
            var remaining = '';
            for (var j = 0; j < r.steps.length; j++) {
                if (!r.steps[j].done && r.steps[j].pct != null) {
                    remaining = 'Collecting ' + r.steps[j].name + ' for ' + (r.symbol || '') + ' — ' + r.steps[j].detail;
                    break;
                }
            }
            infoEl.textContent = remaining || ('Preparing trading system — ' + pct.toFixed(0) + '% complete');
        }
    }

    /* Sparkline history buffers per symbol. Keep last 60 values (~2 min at 2s updates) */
    var _sparkHistory = {}; // { symbol: { rsi: [], macd: [] } }
    var SPARK_MAX = 60;

    function _pushSpark(symbol, key, value) {
        if (value == null || isNaN(value)) return;
        if (!_sparkHistory[symbol]) _sparkHistory[symbol] = {};
        if (!_sparkHistory[symbol][key]) _sparkHistory[symbol][key] = [];
        var arr = _sparkHistory[symbol][key];
        /* Skip duplicate consecutive values — avoids false "movement" at rest */
        if (arr.length && arr[arr.length - 1] === value) return;
        arr.push(value);
        if (arr.length > SPARK_MAX) arr.shift();
    }

    function _renderSparkline(svgId, values, opts) {
        var svg = document.getElementById(svgId);
        if (!svg) return;
        opts = opts || {};
        if (!values || values.length < 2) {
            svg.innerHTML = '';
            return;
        }
        var W = 100, H = 22;
        var min = opts.min != null ? opts.min : Math.min.apply(null, values);
        var max = opts.max != null ? opts.max : Math.max.apply(null, values);
        if (max === min) max = min + 1;

        var n = values.length;
        var dx = W / (n - 1);
        var points = values.map(function(v, i) {
            var x = i * dx;
            var y = H - ((v - min) / (max - min)) * (H - 2) - 1;
            return [x, y];
        });

        var pathD = 'M ' + points.map(function(p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L ');
        var last = points[points.length - 1];

        /* Colour stroke by last vs first */
        var lastVal = values[values.length - 1];
        var firstVal = values[0];
        var color = opts.color || (lastVal > firstVal ? '#4ade80' : lastVal < firstVal ? '#ef4444' : '#818cf8');

        var zoneRect = '';
        if (opts.overbought != null && opts.oversold != null) {
            var obY = H - ((opts.overbought - min) / (max - min)) * (H - 2) - 1;
            var osY = H - ((opts.oversold - min) / (max - min)) * (H - 2) - 1;
            zoneRect = '<rect class="sparkline-zone" x="0" y="0" width="' + W + '" height="' + Math.max(obY, 0).toFixed(1) + '"/>' +
                       '<rect class="sparkline-zone" x="0" y="' + Math.min(osY, H).toFixed(1) + '" width="' + W + '" height="' + (H - Math.min(osY, H)).toFixed(1) + '"/>';
        }

        svg.innerHTML =
            zoneRect +
            '<path class="sparkline-path" d="' + pathD + '" style="stroke:' + color + ';"/>' +
            '<circle class="sparkline-dot" cx="' + last[0].toFixed(1) + '" cy="' + last[1].toFixed(1) + '" r="1.8" style="fill:' + color + ';"/>';
    }

    function updateIndicators(ind, winRate, tradesToday) {
        if (!ind || !ind.symbol) {
            // Нет данных — оставляем "—"
            return;
        }

        // Trend badge — элементы индикаторов существуют не на всех страницах
        var tb = document.getElementById('trend-badge');
        if (!tb) return;

        // Накапливаем историю для sparklines
        _pushSpark(ind.symbol, 'rsi', ind.rsi_14);
        _pushSpark(ind.symbol, 'macd', ind.macd);
        var hist = _sparkHistory[ind.symbol] || {};
        _renderSparkline('spark-rsi', hist.rsi, { min: 0, max: 100, overbought: 70, oversold: 30 });
        _renderSparkline('spark-macd', hist.macd, {});
        var trend = ind.trend || 'neutral';
        var arrows = { bullish: '↑', bearish: '↓', neutral: '→' };
        tb.innerHTML = '<span class="trend-arrow">' + (arrows[trend] || '→') + '</span> ' + escapeHtml(trend.toUpperCase());
        tb.className = 'trend-badge trend-badge-' + trend;

        // EMA
        document.getElementById('ind-ema-short').textContent = '$' + Number(ind.ema_9 || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('ind-ema-long').textContent = 'EMA 21: $' + Number(ind.ema_21 || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});

        // RSI
        var rsi = ind.rsi_14 || 0;
        var rsiEl = document.getElementById('ind-rsi');
        rsiEl.textContent = rsi.toFixed(1);
        var rsiColor = rsi > 70 ? 'var(--red)' : rsi < 30 ? 'var(--green)' : 'var(--accent)';
        rsiEl.style.color = rsiColor;
        var rsiBar = document.getElementById('ind-rsi-bar');
        rsiBar.style.width = rsi + '%';
        rsiBar.style.background = rsiColor;
        var rsiZone = document.getElementById('ind-rsi-zone');
        rsiZone.textContent = rsi > 70 ? 'Перекуплен' : rsi < 30 ? 'Перепродан' : 'Нейтральная зона';
        rsiZone.style.color = rsiColor;

        // MACD
        var macdVal = ind.macd || 0;
        var macdEl = document.getElementById('ind-macd');
        macdEl.textContent = macdVal.toFixed(4);
        macdEl.className = 'ind-card-value ' + (macdVal > 0 ? 'positive' : macdVal < 0 ? 'negative' : 'neutral');
        document.getElementById('ind-macd-signal').textContent = 'Signal: ' + (ind.macd_signal || 0).toFixed(4) + ' | Hist: ' + (ind.macd_histogram || 0).toFixed(4);

        // ADX
        var adxVal = ind.adx || 0;
        document.getElementById('ind-adx').textContent = adxVal.toFixed(1);
        var strengthMap = { strong: 'Сильный тренд', moderate: 'Средний тренд', weak: 'Слабый тренд' };
        var strengthEl = document.getElementById('ind-adx-strength');
        strengthEl.textContent = 'Сила: ' + (strengthMap[ind.trend_strength] || '—');
        strengthEl.style.color = adxVal >= 40 ? 'var(--green)' : adxVal >= 25 ? 'var(--amber)' : 'var(--text-muted)';

        // Bollinger Bands
        document.getElementById('ind-bb-mid').textContent = '$' + Number(ind.bb_middle || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        document.getElementById('ind-bb-range').textContent = '↑ $' + Number(ind.bb_upper || 0).toLocaleString(undefined, {maximumFractionDigits: 2}) + ' / ↓ $' + Number(ind.bb_lower || 0).toLocaleString(undefined, {maximumFractionDigits: 2});

        // ATR
        document.getElementById('ind-atr').textContent = '$' + Number(ind.atr || 0).toFixed(2);
        document.getElementById('ind-bw').textContent = 'Bandwidth: ' + ((ind.bb_bandwidth || 0) * 100).toFixed(2) + '%';

        // Stoch RSI
        var stochRsi = ind.stoch_rsi || 0;
        var stochEl = document.getElementById('ind-stoch-rsi');
        stochEl.textContent = stochRsi.toFixed(1);
        var stochColor = stochRsi > 80 ? 'var(--red)' : stochRsi < 20 ? 'var(--green)' : 'var(--accent-light)';
        stochEl.style.color = stochColor;
        var stochZone = document.getElementById('ind-stoch-zone');
        stochZone.textContent = stochRsi > 80 ? 'Перекуплен' : stochRsi < 20 ? 'Перепродан' : 'Нейтральная зона';
        stochZone.style.color = stochColor;

        // Win Rate Ring
        var circumference = 2 * Math.PI * 34; // r=34
        var offset = circumference - (winRate / 100) * circumference;
        var ringFill = document.getElementById('wr-ring-fill');
        ringFill.style.strokeDashoffset = offset;
        var ringColor = winRate >= 55 ? 'var(--green)' : winRate >= 40 ? 'var(--amber)' : winRate > 0 ? 'var(--red)' : 'var(--accent)';
        ringFill.style.stroke = ringColor;
        var ringText = document.getElementById('wr-ring-text');
        ringText.textContent = winRate.toFixed(1) + '%';
        ringText.className = 'accuracy-ring-text ' + (winRate >= 55 ? 'positive' : winRate >= 40 ? 'warning-text' : winRate > 0 ? 'negative' : 'neutral');
        document.getElementById('wr-trades-count').textContent = tradesToday;
        document.getElementById('wr-trend-label').textContent = trend === 'bullish' ? 'Бычий' : trend === 'bearish' ? 'Медвежий' : 'Боковик';
    }

    /* ── WebSocket ────────────────────────────── */
    var ws = null;
    var wsRetry = 0;
    var MAX_RETRY_DELAY = 30000;

    function connectWS() {
        var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(proto + '//' + location.host + '/ws');

        ws.onopen = function() {
            wsRetry = 0;
            document.getElementById('conn-dot').className = 'conn-dot connected';
            document.getElementById('conn-label').textContent = 'Онлайн';
            var fc = document.getElementById('footer-conn');
            fc.textContent = 'Подключено';
            fc.className = 'footer-dot';
            setBackendOnline(true);
        };

        ws.onmessage = function(e) {
            try {
                var msg = JSON.parse(e.data);
                if (msg.type === 'state_update') updateUI(msg.data);
            } catch(err) {
                _logError('ws.onmessage.parse', err, { raw: String(e.data).slice(0, 200) });
            }
        };

        ws.onclose = function() {
            document.getElementById('conn-dot').className = 'conn-dot disconnected';
            document.getElementById('conn-label').textContent = 'Офлайн';
            var fc = document.getElementById('footer-conn');
            fc.textContent = 'Нет соединения';
            fc.className = 'footer-dot offline';
            setBackendOnline(false);
            var delay = Math.min(1000 * Math.pow(2, wsRetry), MAX_RETRY_DELAY);
            /* Cap at 10 so Math.pow(2, wsRetry) can't overflow on long outages */
            if (wsRetry < 10) wsRetry++;
            setTimeout(connectWS, delay);
        };

        ws.onerror = function() { ws.close(); };
    }
    connectWS();

    /* ── REST Polling ────────────────────────── */
    var _backendOnline = true;
    function setBackendOnline(online) {
        _backendOnline = online;
        var btnStart = document.getElementById('btn-start');
        var btnStop = document.getElementById('btn-stop');
        var btnKill = document.getElementById('btn-kill');
        /* Кнопки управления есть только на index.html */
        if (!btnStart || !btnStop || !btnKill) {
            if (online) fetchStatus();
            return;
        }
        if (!online) {
            btnStart.disabled = true; btnStart.style.opacity = '0.5';
            btnStop.disabled = true; btnStop.style.opacity = '0.5';
            btnKill.disabled = true; btnKill.style.opacity = '0.5';
        } else {
            btnKill.disabled = false; btnKill.style.opacity = '1';
            fetchStatus();
        }
    }

    async function fetchStatus() {
        try {
            var r = await fetch('/api/status');
            var data = await r.json();
            updateUI(data);
            if (!_backendOnline) setBackendOnline(true);
        } catch(e) {
            _logError('fetchStatus', e);
            if (_backendOnline) setBackendOnline(false);
        }
    }

    /* parseOpenedAt + formatDuration live in lib/time.js (unit-tested).
       These bindings stay so the rest of the file can keep calling them
       without a noisy namespace prefix. */
    var _TIME = (typeof SENTINEL !== 'undefined' && SENTINEL.time) || {};
    var parseOpenedAt = _TIME.parseOpenedAt;
    var formatDuration = _TIME.formatDuration;

    function formatOpenTime(val) {
        var d = parseOpenedAt(val);
        if (!d) return '';
        var parts = new Intl.DateTimeFormat('ru-RU', {
            timeZone: 'Asia/Dubai',
            day: '2-digit', month: '2-digit', year: 'numeric',
            hour: '2-digit', minute: '2-digit', hour12: false
        }).formatToParts(d).reduce(function(a,p){ a[p.type]=p.value; return a; }, {});
        var monIdx = parseInt(parts.month, 10) - 1;
        var mon = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек'][monIdx];
        return parts.day + ' ' + mon + ' ' + parts.hour + ':' + parts.minute;
    }

    function buildPositionCard(p) {
        var pnl = p.unrealized_pnl || 0;
        var pnlPct = p.pnl_pct || 0;
        var isLong = (p.side === 'BUY' || p.side === 'LONG');
        var sideClass = isLong ? 'long' : 'short';
        var sideLabel = isLong ? 'BUY (LONG)' : 'SELL (SHORT)';
        var sl = p.stop_loss_price > 0 ? formatUsd(p.stop_loss_price) : '\u2014';
        var tp = p.take_profit_price > 0 ? formatUsd(p.take_profit_price) : '\u2014';
        var rr = p.rr_ratio > 0 ? ('1:' + p.rr_ratio) : '\u2014';
        var progress = p.sl_tp_progress || 0;
        var progressClamped = Math.max(-100, Math.min(100, progress));
        var duration = formatDuration(p.opened_at);
        var openTime = formatOpenTime(p.opened_at);
        var notional = p.notional || 0;

        // Progress bar positioning: center = entry, right = TP, left = SL
        var barWidth = Math.abs(progressClamped) / 2; // max 50% (half of track)
        var barClass = progressClamped >= 0 ? 'profit' : 'loss';
        var barStyle = '';
        if (progressClamped >= 0) {
            barStyle = 'left:50%;width:' + barWidth + '%';
        } else {
            barStyle = 'left:' + (50 - barWidth) + '%;width:' + barWidth + '%';
        }

        var signalHtml = '';
        if (p.signal_reason) {
            signalHtml = '<div class="pos-signal-tag">' +
                '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg>' +
                escapeHtml(p.signal_reason) + '</div>';
        }

        var paperBadge = p.is_paper ? '<span class="pos-paper-badge">PAPER</span>' : '';
        var closeBtn = '<button type="button" class="pos-close-btn" data-symbol="' + escapeHtml(p.symbol) +
            '" onclick="closePosition(this.dataset.symbol)" title="Закрыть эту позицию вручную">' +
            '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" style="width:11px;height:11px;"><path d="M18 6L6 18M6 6l12 12"/></svg>' +
            'Закрыть</button>';

        var sideArrowSvg = isLong
            ? '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:13px;height:13px;"><polyline points="18 15 12 9 6 15"/></svg>'
            : '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:13px;height:13px;"><polyline points="6 9 12 15 18 9"/></svg>';

        return '<div class="pos-card pos-' + sideClass + '">' +
            '<div class="pos-card-top">' +
                _symBadgeHtml(p.symbol) +
                '<span class="pos-side-badge ' + sideClass + '">' + sideArrowSvg + sideLabel + '</span>' +
                '<span class="pos-strategy">' + escapeHtml(p.strategy_name || '\u2014') + '</span>' +
                paperBadge +
                '<div class="pos-meta-right">' +
                    (duration || openTime ? '<span class="pos-duration" title="Открыта: ' + openTime + '"><svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>' + (openTime ? openTime + (duration ? ' (' + duration + ')' : '') : duration) + '</span>' : '') +
                    '<div class="pos-pnl-group">' +
                        '<div class="pos-pnl-value ' + pnlClass(pnl) + '">' + formatPnl(pnl) + '</div>' +
                        '<div class="pos-pnl-pct ' + pnlClass(pnl) + '">' + (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2) + '%</div>' +
                    '</div>' +
                    closeBtn +
                '</div>' +
            '</div>' +
            '<div class="pos-data-grid">' +
                '<div class="pos-data-cell"><div class="pos-data-label">Вход</div><div class="pos-data-value highlight">' + formatUsd(p.entry_price || 0) + '</div></div>' +
                '<div class="pos-data-cell"><div class="pos-data-label">Текущая</div><div class="pos-data-value ' + pnlClass(pnl) + '">' + formatUsd(p.current_price || 0) + '</div></div>' +
                '<div class="pos-data-cell"><div class="pos-data-label">Stop Loss</div><div class="pos-data-value" style="color:var(--red);">' + sl + '</div></div>' +
                '<div class="pos-data-cell"><div class="pos-data-label">Take Profit</div><div class="pos-data-value" style="color:var(--green-bright);">' + tp + '</div></div>' +
                '<div class="pos-data-cell"><div class="pos-data-label">Объём / Нотион.</div><div class="pos-data-value">' + Number(p.quantity || 0).toFixed(6) + '<span style="color:var(--text-dim);font-size:10px;margin-left:4px;">$' + notional.toFixed(0) + '</span></div></div>' +
            '</div>' +
            '<div class="pos-progress-row">' +
                '<div style="flex:1">' +
                    '<div class="pos-progress-labels">' +
                        '<span class="pos-sl-label">SL ' + sl + '</span>' +
                        '<span class="pos-rr-label">R:R ' + rr + '</span>' +
                        '<span class="pos-tp-label">TP ' + tp + '</span>' +
                    '</div>' +
                    '<div class="pos-progress-track" style="margin-top:4px;">' +
                        '<div class="pos-progress-center"></div>' +
                        '<div class="pos-progress-fill ' + barClass + '" style="' + barStyle + '"></div>' +
                    '</div>' +
                '</div>' +
            '</div>' +
            signalHtml +
        '</div>';
    }

    var _fetchingPositions = false;
    async function fetchPositions() {
        if (_fetchingPositions) return;
        _fetchingPositions = true;
        try {
            var r = await fetch('/api/positions');
            var allPositions = await r.json();
            /* Cache full list (unfiltered) for chart overlay + re-apply price lines */
            _openPositions = Array.isArray(allPositions) ? allPositions : [];
            _applyPositionOverlays();
            /* Apply symbol filter */
            var data = _posTradeFilter ? allPositions.filter(function(p) { return p.symbol === _posTradeFilter; }) : allPositions;
            var container = document.getElementById('positions-container');
            var countBadge = document.getElementById('positions-count');
            var totalPnlEl = document.getElementById('positions-total-pnl');
            if (!container || !countBadge || !totalPnlEl) return;
            /* First successful fetch: drop skeleton shimmer */
            if (container.dataset.loading === 'true') container.dataset.loading = 'false';

            /* Build symbol filter bar if multiple symbols */
            _buildSymFilterBar('positions-panel-wrap');

            // Update count badge
            var cnt = data.length;
            var totalAll = allPositions.length;
            var countText = cnt + ' ' + (cnt === 1 ? 'позиция' : cnt < 5 ? 'позиции' : 'позиций');
            if (_posTradeFilter && totalAll !== cnt) countText += ' из ' + totalAll;
            countBadge.textContent = countText;

            if (!cnt) {
                var emptyTitle = _posTradeFilter
                    ? 'Нет позиций по ' + _posTradeFilter.replace('USDT', '')
                    : 'Нет открытых позиций';
                var emptySub = _posTradeFilter
                    ? 'Выберите «Все» чтобы увидеть позиции по всем парам'
                    : 'Когда бот откроет сделку, здесь появится карточка с ценами, PnL и прогрессом до TP/SL';
                container.innerHTML =
                    '<div class="positions-empty">' +
                        '<div class="positions-empty-icon">' +
                            '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>' +
                        '</div>' +
                        '<div class="positions-empty-title">' + escapeHtml(emptyTitle) + '</div>' +
                        '<div class="positions-empty-sub">' + escapeHtml(emptySub) + '</div>' +
                    '</div>';
                totalPnlEl.innerHTML = '<span class="pnl-label">Общий PnL</span><span class="neutral">$0.00</span>';
                return;
            }

            // Total PnL
            var totalPnl = data.reduce(function(s, p) { return s + (p.unrealized_pnl || 0); }, 0);
            totalPnlEl.innerHTML = '<span class="pnl-label">Общий PnL</span><span class="' + pnlClass(totalPnl) + '">' + formatPnl(totalPnl) + '</span>';

            // Render cards
            container.innerHTML = data.map(buildPositionCard).join('');
        } catch(e) {
            _logError('fetchPositions', e);
        } finally {
            _fetchingPositions = false;
        }
    }

    async function fetchTrades() {
        /* Skip entirely if the trades table isn't on this page (e.g. dashboard) */
        var tbody = document.getElementById('trades-table');
        if (!tbody) return;
        try {
            var r = await fetch('/api/trades');
            var allTrades = await r.json();
            /* Apply symbol filter */
            var data = _posTradeFilter ? allTrades.filter(function(t) { return t.symbol === _posTradeFilter; }) : allTrades;
            if (tbody.dataset.loading === 'true') tbody.dataset.loading = 'false';

            /* Build symbol filter bar */
            _buildSymFilterBar('trades-panel-wrap');

            if (!data.length) {
                var msg = _posTradeFilter ? ('Нет сделок по ' + _posTradeFilter.replace('USDT', '')) : 'Сделок пока нет';
                tbody.innerHTML = '<tr><td colspan="6" class="empty-state"><svg aria-hidden="true" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/></svg><div class="empty-state-text">' + escapeHtml(msg) + '</div></td></tr>';
                return;
            }
            tbody.innerHTML = data.slice(0, 20).map(function(t) {
                var pnl = t.pnl || 0;
                var isBuy = t.side === 'BUY';
                var sideClass = isBuy ? 'badge-live' : 'badge-stop';
                var sideArrow = isBuy
                    ? '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:12px;height:12px;"><polyline points="18 15 12 9 6 15"/></svg>'
                    : '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:12px;height:12px;"><polyline points="6 9 12 15 18 9"/></svg>';
                return '<tr>' +
                    '<td class="td-mono">' + escapeHtml(t.time || '-') + '</td>' +
                    '<td>' + _symBadgeHtml(t.symbol) + '</td>' +
                    '<td><span class="badge ' + sideClass + '" style="font-size:11px;padding:3px 10px;gap:4px;">' + sideArrow + escapeHtml(t.side || '-') + '</span></td>' +
                    '<td class="td-mono">' + formatUsd(t.price || 0) + '</td>' +
                    '<td><span class="table-cell-meta" title="' + escapeHtml(t.signal_reason || '') + '">' + escapeHtml(t.signal_reason || '-') + '</span></td>' +
                    '<td class="td-mono ' + pnlClass(pnl) + '" style="font-weight:700;">' + (t.pnl != null ? formatPnl(pnl) : '\u2014') + '</td></tr>';
            }).join('');
        } catch(e) {
            _logError('fetchTrades', e);
        }
    }

    /* PnL-history fallback removed in pro build — the main chart is always the
       market chart. PnL is visible on Equity Curve below. This shim preserves
       the call-site but just renders the current market window. */
    async function fetchPnlHistory() {
        try {
            await renderMarketSeries(activeInterval);
        } catch(e) {
            _logError('fetchPnlHistory', e);
        }
    }


    /* ── Equity Curve Chart ────────────────────── */
    var _equityChart = null;
    function renderEquityCurve(data) {
        var canvas = document.getElementById('equityChart');
        var noDataEl = document.getElementById('equityNoData');
        var skelEl = document.getElementById('equitySkeleton');
        if (!canvas || !data || data.length < 2) return;
        if (noDataEl) noDataEl.style.display = 'none';
        if (skelEl) skelEl.style.display = 'none';

        var labels = data.map(function(d) { return d.label || ''; });
        var balances = data.map(function(d) { return d.balance || 0; });
        // Calculate drawdown series from balance
        var peak = balances[0];
        var drawdowns = balances.map(function(b) {
            if (b > peak) peak = b;
            return peak > 0 ? -((peak - b) / peak * 100) : 0;
        });

        /* Update existing chart instead of destroy/recreate to avoid memory leaks */
        if (_equityChart) {
            _equityChart.data.labels = labels;
            _equityChart.data.datasets[0].data = balances;
            _equityChart.data.datasets[1].data = drawdowns;
            _equityChart.update('none');
            return;
        }
        var ctx = canvas.getContext('2d');

        _equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Balance ($)',
                        data: balances,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.08)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 2,
                        yAxisID: 'y',
                    },
                    {
                        label: 'Drawdown (%)',
                        data: drawdowns,
                        borderColor: 'rgba(239, 68, 68, 0.6)',
                        backgroundColor: 'rgba(239, 68, 68, 0.12)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 1.5,
                        yAxisID: 'y1',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { color: '#8892a8', font: { size: 11 }, boxWidth: 12, padding: 15, usePointStyle: true }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 15, 28, 0.92)',
                        titleColor: '#f5f7ff',
                        bodyColor: '#b8c0d4',
                        borderColor: 'rgba(90, 110, 150, 0.3)',
                        borderWidth: 1,
                        callbacks: {
                            label: function(ctx) {
                                if (ctx.datasetIndex === 0) return 'Balance: $' + ctx.raw.toFixed(2);
                                return 'Drawdown: ' + ctx.raw.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        ticks: { color: '#6a7590', font: { size: 10 }, maxTicksLimit: 8 },
                        grid: { color: 'rgba(56, 68, 91, 0.15)' }
                    },
                    y: {
                        position: 'left',
                        ticks: { color: '#10b981', font: { size: 10 }, callback: function(v) { return '$' + v.toFixed(0); } },
                        grid: { color: 'rgba(56, 68, 91, 0.15)' }
                    },
                    y1: {
                        position: 'right',
                        ticks: { color: '#ef4444', font: { size: 10 }, callback: function(v) { return v.toFixed(0) + '%'; } },
                        grid: { display: false },
                        max: 0,
                    }
                }
            }
        });
    }

    async function fetchEquityCurve() {
        try {
            var r = await fetch('/api/pnl-history');
            if (!r.ok) { _logError('fetchEquityCurve', 'non-200', { status: r.status }); return; }
            var data = await r.json();
            if (data && data.length >= 2) renderEquityCurve(data);
        } catch(e) {
            _logError('fetchEquityCurve', e);
        }
    }

    /* ── News & Sentiment — PRO MAX ────────────────────── */
    function fgiColor(val) {
        if (val <= 20) return '#ef4444';
        if (val <= 40) return '#f59e0b';
        if (val <= 60) return '#8892a8';
        if (val <= 80) return '#10b981';
        return '#4ade80';
    }
    function fgiLabel(val) {
        if (val <= 20) return 'Сильный страх';
        if (val <= 40) return 'Страх';
        if (val <= 60) return 'Нейтрально';
        if (val <= 80) return 'Жадность';
        return 'Сильная жадность';
    }
    function timeAgo(ts) {
        var diff = Math.floor(Date.now() / 1000) - ts;
        if (diff < 60) return 'now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h';
        return Math.floor(diff / 86400) + 'd';
    }

    var _newsActiveFilter = 'all';
    var _newsData = [];

    function _renderOneCard(n) {
        var eff = n.effective_impact || 0;
        var abs = Math.abs(eff);
        var cls = eff > 0.05 ? 'bull' : eff < -0.05 ? 'bear' : 'flat';
        var icCls = eff > 0.05 ? 'ic-bull' : eff < -0.05 ? 'ic-bear' : 'ic-flat';
        var arrow = eff > 0.05 ? '&#9650;' : eff < -0.05 ? '&#9660;' : '';
        var sign = eff >= 0 ? '+' : '';

        var titleHtml = n.url
            ? '<a href="' + escapeHtml(n.url) + '" target="_blank" rel="noopener">' + escapeHtml(n.title) + '</a>'
            : escapeHtml(n.title);

        var reasonHtml = n.llm_reasoning
            ? '<div class="news-reasoning">' + escapeHtml(n.llm_reasoning) + '</div>' : '';

        var modeTag = n.analysis_mode === 'llm'
            ? '<span class="news-mode-tag llm">AI</span>'
            : '<span class="news-mode-tag kw">KW</span>';

        var urgHtml = '';
        if (n.urgency && n.urgency !== 'low') {
            urgHtml = '<span class="news-urgency-badge ' + escapeHtml(n.urgency) + '">' + escapeHtml(n.urgency) + '</span>';
        }

        var catHtml = '';
        if (n.category && n.category !== 'other') {
            catHtml = '<span class="news-cat-tag">' + escapeHtml(n.category) + '</span>';
        }

        var coinsHtml = '';
        if (n.coins_mentioned && n.coins_mentioned.length) {
            coinsHtml = n.coins_mentioned.slice(0, 3).map(function(c) {
                return '<span class="news-coin-tag">' + escapeHtml(c) + '</span>';
            }).join('');
        }

        var conf = n.confidence || 0;
        var confColor = conf > 0.7 ? 'var(--green)' : conf > 0.4 ? 'var(--amber)' : 'var(--red)';
        var confHtml = n.analysis_mode === 'llm'
            ? '<span class="news-conf-inline"><span class="news-conf-track"><span class="news-conf-fill" style="width:' + (conf*100) + '%;background:' + confColor + '"></span></span><span class="news-conf-pct" style="color:' + confColor + '">' + (conf*100).toFixed(0) + '%</span></span>'
            : '';

        return '<div class="news-card">' +
            '<div class="news-card-stripe s-' + cls + '"></div>' +
            '<div class="news-card-body">' +
                '<div class="news-card-top">' +
                    '<span class="news-impact-chip ' + icCls + '">' + arrow + ' ' + sign + abs.toFixed(2) + '%</span>' +
                    '<div class="news-card-title">' + titleHtml + '</div>' +
                '</div>' +
                reasonHtml +
                '<div class="news-card-meta">' +
                    modeTag + urgHtml + catHtml +
                    '<span class="news-src">' + escapeHtml(n.source || '') + '</span>' +
                    '<span class="news-dot">&middot;</span>' +
                    '<span class="news-time">' + timeAgo(n.published_at || 0) + '</span>' +
                    confHtml + coinsHtml +
                '</div>' +
            '</div>' +
        '</div>';
    }

    function renderNewsCards(news, filter) {
        var filtered = filter === 'all' ? news : news.filter(function(n) {
            return (n.category || 'other') === filter;
        });

        // Разделяем: позитив и негатив (нейтрал ~0 — скрыт)
        var bullish = filtered.filter(function(n) { return (n.effective_impact || 0) > 0.05; });
        var bearish = filtered.filter(function(n) { return (n.effective_impact || 0) < -0.05; });

        // Сортируем по силе влияния (самые сильные сверху)
        bullish.sort(function(a, b) { return (b.effective_impact || 0) - (a.effective_impact || 0); });
        bearish.sort(function(a, b) { return (a.effective_impact || 0) - (b.effective_impact || 0); });

        // Счётчики в заголовках
        document.getElementById('news-bull-hdr-count').textContent = bullish.length;
        document.getElementById('news-bear-hdr-count').textContent = bearish.length;

        // Левая колонка — позитив
        var bullEl = document.getElementById('news-list-bull');
        if (bullish.length) {
            bullEl.innerHTML = bullish.map(_renderOneCard).join('');
        } else {
            bullEl.innerHTML = '<div class="news-empty"><div>Нет позитивных новостей</div></div>';
        }

        // Правая колонка — негатив
        var bearEl = document.getElementById('news-list-bear');
        if (bearish.length) {
            bearEl.innerHTML = bearish.map(_renderOneCard).join('');
        } else {
            bearEl.innerHTML = '<div class="news-empty"><div>Нет негативных новостей</div></div>';
        }
    }

    var _catNames = {
        'all': 'Все', 'macro': 'Макро', 'regulation': 'Регуляция',
        'adoption': 'Принятие', 'technology': 'Технологии', 'market': 'Рынок',
        'defi': 'DeFi', 'security': 'Безопасн.', 'mining': 'Майнинг',
        'nft': 'NFT', 'other': 'Другое', 'institutional': 'Институц.',
        'exchange': 'Биржи', 'stablecoin': 'Стейблкоины', 'legal': 'Право',
    };
    function buildFilterTabs(news) {
        var counts = { all: news.length };
        news.forEach(function(n) {
            var cat = n.category || 'other';
            counts[cat] = (counts[cat] || 0) + 1;
        });
        var row = document.getElementById('news-filter-row');
        var cats = Object.keys(counts).sort(function(a, b) {
            if (a === 'all') return -1; if (b === 'all') return 1;
            return counts[b] - counts[a];
        });
        row.innerHTML = cats.slice(0, 8).map(function(cat) {
            var active = cat === _newsActiveFilter ? ' active' : '';
            var label = _catNames[cat] || cat;
            return '<button class="news-filter-btn' + active + '" data-cat="' + escapeHtml(cat) + '">' +
                escapeHtml(label) + '<span class="news-filter-count">' + counts[cat] + '</span></button>';
        }).join('');
        row.querySelectorAll('.news-filter-btn').forEach(function(btn) {
            btn.onclick = function() {
                _newsActiveFilter = btn.dataset.cat;
                row.querySelectorAll('.news-filter-btn').forEach(function(b) { b.classList.remove('active'); });
                btn.classList.add('active');
                renderNewsCards(_newsData, _newsActiveFilter);
            };
        });
    }

    async function fetchNews() {
        /* Skip entirely if the news panel isn't on this page (e.g. dashboard) */
        var fgiEl = document.getElementById('news-fgi-value');
        if (!fgiEl) return;
        try {
            var r = await fetch('/api/news');
            var data = await r.json();
            var sentiment = data.sentiment || {};
            var impact = data.impact || {};
            var news = data.news || [];
            var signal = data.signal || {};
            _newsData = news;

            // ── Gauge ──
            var fgi = sentiment.fear_greed_index || 50;
            fgiEl.textContent = fgi;
            fgiEl.style.color = fgiColor(fgi);
            document.getElementById('news-fgi-label').textContent = fgiLabel(fgi);

            // Needle rotation: 0→-90deg, 100→+90deg
            var needle = document.getElementById('news-gauge-needle');
            if (needle) {
                var angle = -90 + (fgi / 100) * 180;
                needle.setAttribute('transform', 'rotate(' + angle + ' 60 58)');
            }

            // ── Stats ──
            var bull = sentiment.bullish_count || 0;
            var bear = sentiment.bearish_count || 0;
            var neut = sentiment.neutral_count || 0;
            document.getElementById('news-bull-count').textContent = bull;
            document.getElementById('news-bear-count').textContent = bear;
            document.getElementById('news-neutral-count').textContent = neut;

            var avgImpact = impact.avg_effective_impact || impact.avg_impact_pct || 0;
            var avgEl = document.getElementById('news-avg-impact');
            avgEl.textContent = (avgImpact >= 0 ? '+' : '') + avgImpact.toFixed(2) + '%';
            avgEl.style.color = avgImpact > 0.15 ? 'var(--green)' : avgImpact < -0.15 ? '#f87171' : 'var(--cyan)';

            var conVal = impact.consensus_strength || 0;
            var consensusEl = document.getElementById('news-consensus-val');
            consensusEl.textContent = conVal.toFixed(0) + '%';
            consensusEl.style.color = conVal > 60 ? 'var(--green)' : 'var(--purple)';

            document.getElementById('news-critical-count').textContent = impact.critical_count || 0;

            // ── Pulse bar ──
            var total = bull + bear + neut || 1;
            document.getElementById('news-pulse-bull').style.width = (bull / total * 100) + '%';
            document.getElementById('news-pulse-neut').style.width = (neut / total * 100) + '%';
            document.getElementById('news-pulse-bear').style.width = (bear / total * 100) + '%';

            // ── Direction badge ──
            var dir = impact.overall_direction || 'neutral';
            var badge = document.getElementById('news-direction-badge');
            badge.className = 'news-direction-badge ' + dir;
            badge.textContent = dir === 'bullish' ? 'РОСТ' : dir === 'bearish' ? 'ПАДЕНИЕ' : 'НЕЙТРАЛ';

            // ── Signal strip ──
            var sigBar = document.getElementById('news-signal-bar');
            if (signal.composite_score !== undefined) {
                sigBar.style.display = 'flex';
                var sc = signal.composite_score || 0;
                var scoreEl = document.getElementById('news-signal-score');
                scoreEl.textContent = (sc >= 0 ? '+' : '') + sc.toFixed(3);
                scoreEl.style.color = sc > 0.1 ? 'var(--green-bright)' : sc < -0.1 ? '#f87171' : 'var(--text-muted)';

                document.getElementById('news-signal-strength').textContent = ((signal.signal_strength || 0) * 100).toFixed(0) + '%';
                document.getElementById('news-signal-strength').style.color = (signal.signal_strength || 0) > 0.5 ? 'var(--green)' : 'var(--text-secondary)';

                var biasEl = document.getElementById('news-signal-bias');
                var bias = signal.bias || 'neutral';
                var biasLabels = { bullish: 'РОСТ', bearish: 'ПАДЕНИЕ', neutral: 'НЕЙТРАЛ' };
                biasEl.textContent = biasLabels[bias] || bias.toUpperCase();
                biasEl.className = 'nss-bias-pill ' + bias;

                document.getElementById('news-signal-category').textContent = (signal.dominant_category || '—').toUpperCase();

                sigBar.className = 'news-signal-strip' + (signal.critical_alert ? ' critical-glow' : '');
            }

            // ── Filters + cards ──
            if (!news.length) {
                document.getElementById('news-filter-row').innerHTML = '';
                document.getElementById('news-list-bull').innerHTML = '<div class="news-empty"><div>Нет новостей</div></div>';
                document.getElementById('news-list-bear').innerHTML = '<div class="news-empty"><div>Нет новостей</div></div>';
                document.getElementById('news-bull-hdr-count').textContent = '0';
                document.getElementById('news-bear-hdr-count').textContent = '0';
                return;
            }

            buildFilterTabs(news);
            renderNewsCards(news, _newsActiveFilter);
        } catch(e) { _logError('fetch', e); }
    }

    /* ── Control Actions ─────────────────────── */
    async function controlAction(action) {
        if (!_backendOnline) {
            showToast('Backend не запущен — запустите main.py', 'error');
            return;
        }
        var btn = document.getElementById('btn-' + (action === 'resume' ? 'start' : action));
        if (btn) btn.disabled = true;
        try {
            var r = await _csrfFetch('/api/control/' + action, { method: 'POST' });
            var data = await r.json();
            if (r.ok) {
                showToast(
                    action === 'resume' ? 'Торговля запущена' :
                    action === 'stop' ? 'Торговля приостановлена' :
                    'Аварийная остановка выполнена',
                    action === 'kill' ? 'warning' : 'success'
                );
            } else {
                showToast(data.error || 'Действие не выполнено', 'error');
            }
            fetchStatus();
        } catch(e) {
            setBackendOnline(false);
            showToast('Backend не запущен — запустите main.py', 'error');
        }
    }

    /* confirmKill & closePosition — определены через window.* ниже (с кастомной модалью) */

    /* ══════════════════════════════════════════
       STRATEGY PERFORMANCE
       ══════════════════════════════════════════ */

    async function fetchStrategyPerformance() {
        /* Skip entirely if the grid isn't on this page (e.g. dashboard) */
        const grid = document.getElementById('strat-perf-grid');
        if (!grid) return;
        try {
            const res = await fetch('/api/strategy-performance');
            const data = await res.json();
            if (!data || !data.length) {
                grid.innerHTML = '<div class="strat-log-empty"><svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width:32px;height:32px;opacity:0.2;"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg><div>Данных по стратегиям пока нет</div></div>';
                return;
            }
            grid.innerHTML = data.map(s => {
                const pnl = s.total_pnl || 0;
                const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
                const pnlSign = pnl >= 0 ? '+' : '';
                const wr = s.win_rate || 0;
                const wrColor = wr >= 60 ? 'var(--green)' : wr >= 40 ? 'var(--amber)' : 'var(--red)';
                return `
                    <div class="strat-perf-card">
                        <div class="sp-name">
                            <span>${escapeHtml(s.strategy_name || '')}</span>
                            <span class="sp-pnl" style="color:${pnlColor}">${pnlSign}$${pnl.toFixed(2)}</span>
                        </div>
                        <div class="sp-stats">
                            <span>Trades<span class="sp-val">${s.total_trades}</span></span>
                            <span>Win Rate<span class="sp-val" style="color:${wrColor}">${wr.toFixed(0)}%</span></span>
                            <span>PF<span class="sp-val" style="color:${(s.profit_factor||0) >= 1.5 ? 'var(--green)' : (s.profit_factor||0) >= 1.0 ? 'var(--amber)' : 'var(--red)'}">${(s.profit_factor||0).toFixed(2)}</span></span>
                            <span>Avg PnL<span class="sp-val">${(s.avg_pnl||0) >= 0 ? '+' : ''}$${(s.avg_pnl||0).toFixed(2)}</span></span>
                            <span>Best<span class="sp-val" style="color:var(--green)">+$${(s.best_trade||0).toFixed(2)}</span></span>
                            <span>Worst<span class="sp-val" style="color:var(--red)">$${(s.worst_trade||0).toFixed(2)}</span></span>
                        </div>
                        <div class="sp-bar">
                            <div class="sp-bar-fill" style="width:${wr}%;background:${wrColor}"></div>
                        </div>
                    </div>`;
            }).join('');
        } catch(e) { _logError('fetch', e); }
    }

    /* ── Init ──────────────────────────────────
       Centralised poll schedule. Interval IDs are tracked so we can
       clear them on beforeunload — without cleanup, fetches continue
       firing after navigation and leak memory on every page refresh.

       `status` intentionally absent from schedule: the WebSocket pushes
       state every ~2s. REST fetchStatus() fires only via
       `_statusFallbackTick` while the socket is offline.
    */
    const POLL_INTERVALS_MS = {
        status_fallback: 10000, // REST fallback — only runs while WS is down
        positions: 8000,    // open positions table
        trades:    8000,    // recent trades list
        chart:    30000,    // Chart.js market series
        equity:   30000,    // equity curve
        perf:     60000,    // per-strategy performance
        news:    120000,    // news feed (slow API)
    };
    const _activeTimers = [];
    function _schedule(fn, ms) { _activeTimers.push(setInterval(fn, ms)); }

    /* REST status fallback: only polls when the WebSocket is disconnected. */
    function _statusFallbackTick() {
        if (!_backendOnline || !ws || ws.readyState !== 1) fetchStatus();
    }

    fetchStatus();
    fetchPositions();
    fetchTrades();
    fetchPnlHistory();
    fetchEquityCurve();
    fetchStrategyPerformance();
    fetchNews();
    _schedule(_statusFallbackTick,      POLL_INTERVALS_MS.status_fallback);
    _schedule(fetchPositions,           POLL_INTERVALS_MS.positions);
    _schedule(fetchTrades,              POLL_INTERVALS_MS.trades);
    _schedule(function() {
        /* Skip poll while user is browsing historical window — refreshing
           would be a no-op (same frozen data) and would snap zoom / reset
           pan state needlessly. */
        if (_chartEndTs > 0) return;
        renderMarketSeries(activeInterval);
    }, POLL_INTERVALS_MS.chart);
    _schedule(fetchEquityCurve,         POLL_INTERVALS_MS.equity);
    _schedule(fetchStrategyPerformance, POLL_INTERVALS_MS.perf);
    _schedule(fetchNews,                POLL_INTERVALS_MS.news);

    window.addEventListener('beforeunload', () => {
        _activeTimers.forEach(id => clearInterval(id));
        _activeTimers.length = 0;
    });

    /* ══════════════════════════════════════════
       Control buttons — event delegation
       Replaces per-button inline onclick handlers so the markup
       stays CSP-friendly and listeners survive HTML re-renders.
       ══════════════════════════════════════════ */
    (function initControlButtons() {
        var bar = document.querySelector('.controls-bar .controls-actions');
        if (bar) {
            bar.addEventListener('click', function(ev) {
                var btn = ev.target.closest('button[data-action]');
                if (!btn || btn.disabled) return;
                var action = btn.getAttribute('data-action');
                if (action === 'kill') {
                    if (typeof window.confirmKill === 'function') window.confirmKill();
                } else if (action === 'resume' || action === 'stop') {
                    controlAction(action);
                }
            });
        }
        var resetBtn = document.getElementById('chartResetZoomBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', function() {
                if (lwcChart) { try { lwcChart.timeScale().fitContent(); } catch(_e) {} }
                _userHasZoomed = false;
            });
        }
        /* Logout button in header */
        var logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', async function() {
                try {
                    await _csrfFetch('/api/logout', { method: 'POST' });
                } catch (e) {
                    _logError('logout', e);
                }
                /* Navigate regardless — cookie is either gone or never existed */
                window.location.href = '/login';
            });
        }

        /* Chart legend toggles are wired in initLegendToggles() above. */
    })();

    /* Sidebar rendering and mobile-menu wiring live in /static/js/sidebar.js
       (loaded standalone on every page). */

    /* ══════════════════════════════════════════
       Keyboard nav for role=radiogroup bars
       (symbolBar, chartTypeBar, intervalBar)
       ══════════════════════════════════════════ */
    (function initRadioKeyboardNav() {
        document.querySelectorAll('[role="radiogroup"]').forEach(function(group) {
            group.addEventListener('keydown', function(e) {
                if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight' &&
                    e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
                var buttons = Array.prototype.slice.call(group.querySelectorAll('[role="radio"]'));
                if (buttons.length < 2) return;
                var current = buttons.indexOf(document.activeElement);
                if (current === -1) current = buttons.findIndex(function(b) { return b.getAttribute('aria-checked') === 'true'; });
                if (current === -1) current = 0;
                var next = (e.key === 'ArrowLeft' || e.key === 'ArrowUp')
                    ? (current - 1 + buttons.length) % buttons.length
                    : (current + 1) % buttons.length;
                e.preventDefault();
                buttons[next].focus();
                buttons[next].click();
            });
        });
    })();

    /* ══════════════════════════════════════════
       Custom confirmation modal
       Replaces window.confirm() for Emergency Stop
       ══════════════════════════════════════════ */
    function showConfirmModal(opts) {
        return new Promise(function(resolve) {
            var root = document.getElementById('confirm-modal-root');
            if (!root) { resolve(window.confirm(opts.title + '\n\n' + (opts.body || ''))); return; }

            var positionsHtml = '';
            if (opts.positions && opts.positions.length > 0) {
                var rows = opts.positions.map(function(p) {
                    var pnl = p.unrealized_pnl || 0;
                    var pnlCls = pnl >= 0 ? 'positive' : 'negative';
                    var pnlStr = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
                    return '<div class="pos-row"><span>' + escapeHtml(p.symbol || '?') + ' · ' + escapeHtml(p.side || '') + '</span>' +
                           '<span class="' + pnlCls + '">' + pnlStr + '</span></div>';
                }).join('');
                positionsHtml =
                    '<div class="confirm-modal-positions">' +
                    '<div class="confirm-modal-positions-title">Закроются позиции (' + opts.positions.length + '):</div>' +
                    '<div class="confirm-modal-positions-list">' + rows + '</div>' +
                    '</div>';
            }

            var typePhrase = opts.requireTypeConfirm ? String(opts.requireTypeConfirm) : '';
            var typeInputHtml = typePhrase
                ? '<div class="confirm-modal-typelock">' +
                  '  <label class="confirm-modal-typelock-label" for="confirm-modal-type">' +
                  '    Чтобы подтвердить, введите <code>' + escapeHtml(typePhrase) + '</code>' +
                  '  </label>' +
                  '  <input id="confirm-modal-type" class="confirm-modal-typelock-input" type="text" autocomplete="off" spellcheck="false" placeholder="Введите ' + escapeHtml(typePhrase) + '" aria-label="Подтверждение ввода">' +
                  '</div>'
                : '';

            var html =
                '<div class="confirm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="confirm-modal-title">' +
                '  <div class="confirm-modal confirm-modal--warn" tabindex="-1">' +
                '    <div class="confirm-modal-header">' +
                '      <div class="confirm-modal-icon">' +
                '        <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>' +
                '      </div>' +
                '      <div class="confirm-modal-title" id="confirm-modal-title">' + escapeHtml(opts.title) + '</div>' +
                '    </div>' +
                '    <div class="confirm-modal-body">' + (opts.body || '') + positionsHtml + typeInputHtml + '</div>' +
                '    <div class="confirm-modal-actions">' +
                '      <button type="button" class="btn btn-cancel" data-action="cancel">Отмена</button>' +
                '      <button type="button" class="btn btn-danger" data-action="confirm"' + (typePhrase ? ' disabled' : '') + '>' + escapeHtml(opts.confirmText || 'Подтвердить') + '</button>' +
                '    </div>' +
                '  </div>' +
                '</div>';
            root.innerHTML = html;

            var backdrop = root.querySelector('.confirm-modal-backdrop');
            var modal = root.querySelector('.confirm-modal');
            var btnConfirm = root.querySelector('[data-action="confirm"]');
            var btnCancel = root.querySelector('[data-action="cancel"]');
            var typeInput = root.querySelector('#confirm-modal-type');
            var prevFocus = document.activeElement;

            function isPhraseOk() {
                if (!typePhrase) return true;
                return typeInput && typeInput.value.trim().toUpperCase() === typePhrase.toUpperCase();
            }
            function refreshConfirmState() {
                if (!typePhrase) return;
                var ok = isPhraseOk();
                btnConfirm.disabled = !ok;
                if (ok) btnConfirm.classList.add('armed');
                else btnConfirm.classList.remove('armed');
            }

            function close(result) {
                root.innerHTML = '';
                document.removeEventListener('keydown', onKey);
                if (prevFocus && prevFocus.focus) prevFocus.focus();
                resolve(result);
            }
            function tryConfirm() {
                if (!isPhraseOk()) return;
                close(true);
            }
            function onKey(e) {
                if (e.key === 'Escape') { e.preventDefault(); close(false); }
                if (e.key === 'Enter' && document.activeElement !== btnCancel) {
                    e.preventDefault(); tryConfirm();
                }
            }
            btnConfirm.addEventListener('click', tryConfirm);
            btnCancel.addEventListener('click', function() { close(false); });
            backdrop.addEventListener('click', function(e) { if (e.target === backdrop) close(false); });
            document.addEventListener('keydown', onKey);
            if (typeInput) {
                typeInput.addEventListener('input', refreshConfirmState);
                /* For type-lock: focus the input so user can type right away */
                setTimeout(function() { typeInput.focus(); }, 50);
            } else {
                /* Initial focus on Cancel for safety (destructive action) */
                setTimeout(function() { btnCancel.focus(); }, 50);
            }
        });
    }

    /* Override confirmKill & closePosition to use the custom modal */
    window.confirmKill = async function() {
        var positions = [];
        try {
            var r = await fetch('/api/positions');
            if (r.ok) positions = await r.json();
        } catch (_) { /* ignore */ }
        var hasPositions = positions.length > 0;
        var ok = await showConfirmModal({
            title: 'Аварийная остановка',
            body: 'Все открытые позиции будут закрыты <strong>немедленно</strong> по рыночной цене, все активные ордера отменены.' +
                  (hasPositions ? '' : '<br><br>Сейчас нет открытых позиций — команда просто остановит бота.'),
            positions: positions,
            /* Pro-grade safety: require typing STOP when real money is at risk */
            requireTypeConfirm: hasPositions ? 'STOP' : '',
            confirmText: hasPositions ? 'Закрыть ВСЁ' : 'Остановить бота'
        });
        if (ok) controlAction('kill');
    };

    window.closePosition = async function(symbol) {
        if (!symbol) return;
        if (!_backendOnline) {
            showToast('Backend не запущен — запустите main.py', 'error');
            return;
        }
        var ok = await showConfirmModal({
            title: 'Закрыть позицию ' + symbol + '?',
            body: 'Ордер уйдёт на исполнение <strong>немедленно</strong> по рыночной цене.',
            confirmText: 'Закрыть позицию'
        });
        if (!ok) return;
        try {
            var r = await _csrfFetch('/api/positions/' + encodeURIComponent(symbol) + '/close', { method: 'POST' });
            var data = {};
            try { data = await r.json(); } catch (_) { /* ignore */ }
            if (r.ok && data.ok !== false) {
                showToast('Закрываю ' + symbol + '...', 'success');
                fetchPositions();
                fetchStatus();
            } else {
                showToast(data.error || ('Не удалось закрыть ' + symbol), 'error');
            }
        } catch (e) {
            setBackendOnline(false);
            showToast('Ошибка закрытия: ' + (e && e.message ? e.message : e), 'error');
        }
    };

    /* --- Block Info Modals Script --- */
            const blockDescriptions = {
        "PnL Сегодня": "Показывает общую сумму реализованной прибыли или убытка за день.",
        "PnL Всего": "Отражает общий накопленный результат от всех закрытых сделок.",
        "Баланс": "Текущий доступный капитал на счету для сделок.",
        "Позиции": "Количество открытых позиций в данный момент времени.",
        "Сделок сегодня": "Количество закрытых сделок (ордеров) за сегодня.",
        "Win Rate": "Процент прибыльных сделок (Эффективность стратегий).",
        "PnL History": "График изменения капитала во времени.",
        "Positions": "Подробный список всех текущих открытых позиций.",
        "Trades (24h)": "Лента последних завершенных сделок за сутки.",
        "Strategies": "Мониторинг запущенных торговых стратегий.",
        "Event Log": "Журнал внутренних событий системы (логи).",
        "Status": "Общее состояние бота и компонентов.",
        "Crypto News & Market Impact": "Анализ новостей и влияние на курс.",
        "Candle": "Японские свечи — цены открытия, закрытия, максимум и минимум.",
        "Line": "Линейный график — строится по ценам закрытия.",
        "1m": "Интервал 1 минута. Быстрый рынок.",
        "5m": "Интервал 5 минут. Краткосрочный тренд.",
        "15m": "Интервал 15 минут.",
        "1h": "Интервал 1 час.",
        "4h": "Интервал 4 часа. Среднесрочный анализ.",
        "1d": "Дневные свечи. Глобальный тренд.",
        "Монета": "Торговая пара (актив), по которой открыта позиция.",
        "Стратегия": "Название алгоритма, который открыл позицию.",
        "Сторона": "Направление сделки: BUY (Лонг) или SELL (Шорт).",
        "Вход": "Цена, по которой была открыта позиция.",
        "Текущая": "Актуальная рыночная цена актива.",
        "SL / TP": "Уровни фиксации (Stop Loss / Take Profit).",
        "Кол-во": "Объем актива в сделке (штук).",
        "Время": "Время завершения (открытия).",
        "Цена": "Курс закрытия сделки.",
        "Прибыль": "Заработок по сделке в валюте (PnL).",
        "Uptime": "Время работы робота без остановок.",
        "API": "Подключение к бирже.",
        "Database": "Статус БД.",
        "Collector": "Сборщик рыночных данных.",
        "PnL": "Ваш текущий финансовый результат.",
        "Max Drawdown": "Максимальное падение баланса от пикового значения. Показывает наихудший сценарий. Профи: до 10%.",
        "Profit Factor": "Отношение общей прибыли к общим убыткам. >1.5 = сильная система, <1.0 = убыточная.",
        "Avg R:R": "Средний Risk:Reward — отношение среднего выигрыша к среднему проигрышу. >1.5 = профессиональный уровень.",
        "Equity Curve": "Кривая роста капитала с наложением просадок. Идеальная — плавный рост без глубоких провалов."
    };

    function createBlockInfoModal() {
        const modalHtml = `
            <div id="blockInfoModal" class="modal-overlay" style="display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.6); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); align-items: center; justify-content: center; z-index: 10000;">
                <div class="modal-content" style="width: 100%; max-width: 420px; text-align: left; background: rgba(12, 17, 28, 0.92); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border-radius: var(--radius-xl); border: 1px solid var(--border-light); box-shadow: 0 24px 64px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.08); z-index: 10001;">
                    <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; padding: 20px 24px; border-bottom: 1px solid var(--border);">
                        <h3 id="blockInfoTitle" style="margin: 0; font-size: 15px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.01em;">Информация</h3>
                        <button class="modal-close" id="blockInfoClose" style="background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text-muted); width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; cursor: pointer; transition: all 0.2s;">&times;</button>
                    </div>
                    <div class="modal-body" style="padding: 20px 24px 24px;">
                        <p id="blockInfoText" style="color: var(--text-secondary); font-weight: 500; font-size: 13px; line-height: 1.7; margin: 0;"></p>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        document.getElementById('blockInfoClose').addEventListener('click', () => {
            document.getElementById('blockInfoModal').style.display = 'none';
        });

        document.getElementById('blockInfoModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('blockInfoModal')) {
                document.getElementById('blockInfoModal').style.display = 'none';
            }
        });
    }

    function initBlockInfoButtons() {
        createBlockInfoModal();

        const style = document.createElement('style');
        style.innerHTML = `
            .block-info-btn {
                background: var(--accent-dim); border: 1px solid rgba(99,102,241,0.2); color: var(--accent-light); cursor: pointer; padding: 3px; border-radius: 7px; display: inline-flex; align-items: center; justify-content: center; transition: all 0.25s cubic-bezier(0.4,0,0.2,1); margin-left: 8px; opacity: 0.35; width: 20px; height: 20px; outline: none; flex-shrink: 0;
            }
            .metric-card:hover .block-info-btn, .panel:hover .block-info-btn, .table-panel:hover .block-info-btn, .readiness-panel:hover .block-info-btn, .strategy-log-panel:hover .block-info-btn, .activity-panel:hover .block-info-btn, .news-panel:hover .block-info-btn, th:hover .block-info-btn, .chart-type-btn:hover .block-info-btn, .interval-btn:hover .block-info-btn { opacity: 0.7; }
            .block-info-btn:hover { color: #fff; background: var(--accent); border-color: var(--accent); transform: scale(1.1); box-shadow: 0 0 12px rgba(99,102,241,0.35); opacity: 1 !important; }
            .block-info-btn svg { width: 12px; height: 12px; }
        `;
        document.head.appendChild(style);

        const svgIcon = '<svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';

        // Add info button to Metric Cards
        document.querySelectorAll('.metric-card').forEach(card => {
            const header = card.querySelector('.metric-header');
            const labelSpan = card.querySelector('.metric-label');
            if (header && labelSpan) {
                const title = labelSpan.textContent.trim();
                const btn = document.createElement('button');
                btn.className = 'block-info-btn';
                btn.innerHTML = svgIcon;
                btn.title = 'Подробнее';
                btn.onclick = (e) => {
                    e.stopPropagation();
                    showBlockInfo(title);
                };
                header.appendChild(btn);
            }
        });

        // Add info button to Panels
        document.querySelectorAll('.panel').forEach(panel => {
            const titleRow = panel.querySelector('.panel-title');
            if (titleRow) {
                const h3 = titleRow.querySelector('h3');
                if (h3) {
                    const clone = h3.cloneNode(true);
                    clone.querySelectorAll('svg, span').forEach(s => s.remove());
                    let cleanTitle = clone.textContent.trim();

                    const btn = document.createElement('button');
                    btn.className = 'block-info-btn';
                    btn.innerHTML = svgIcon;
                    btn.title = 'Подробнее';
                    btn.style.marginLeft = '12px';
                    btn.onclick = (e) => {
                        e.stopPropagation();
                        showBlockInfo(cleanTitle);
                    };
                    
                    // Insert right after the h3 text and icon
                    h3.appendChild(btn);
                    h3.style.display = 'flex';
                    h3.style.alignItems = 'center';
                }
            }
        });

        // Add info button to table headers (th) and status items
        document.querySelectorAll('th, .status-label, .chart-type-btn, .interval-btn').forEach(el => {
            const title = el.textContent.trim();
            if (blockDescriptions[title]) {
                const btn = document.createElement('button');
                btn.className = 'block-info-btn';
                btn.innerHTML = svgIcon;
                btn.title = 'Подробнее';
                btn.style.marginLeft = '3px';
                btn.style.transform = 'scale(0.75)';
                btn.onclick = (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    showBlockInfo(title);
                };
                el.appendChild(btn);
                el.style.display = 'inline-flex';
                el.style.alignItems = 'center';
            }
        });
    }

    function showBlockInfo(title) {
        let desc = blockDescriptions[title] || "Подробная информация для панели: " + title;
        document.getElementById('blockInfoTitle').textContent = title;
        document.getElementById('blockInfoText').textContent = desc;
        document.getElementById('blockInfoModal').style.display = 'flex';
    }

    setTimeout(initBlockInfoButtons, 500);
    