/**
 * Shared sidebar behaviour for every dashboard page.
 *
 * - Renders the canonical navigation into <nav id="sidebar"> (if empty) so
 *   every page shows the same menu, in the same order, with the same icons.
 * - Marks the link matching window.location.pathname as `.active`.
 * - Wires up the mobile hamburger (#mobile-menu-toggle) + backdrop so the
 *   slide-out drawer works on every page, not just the dashboard.
 *
 * Loaded standalone (no dependencies). Safe to re-include alongside index.js
 * — both are idempotent via dataset flags.
 */
(function () {
    'use strict';

    var SVG_OPEN  = '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2">';
    var SVG_CLOSE = '</svg>';

    var NAV_GROUPS = [
        {
            items: [
                { href: '/',              label: 'Dashboard',      icon: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>' },
                { href: '/analytics',     label: 'Аналитика',      icon: '<path d="M3 3v18h18"/><path d="M7 14l4-4 4 4 5-6"/>' },
                { href: '/ml-robustness', label: 'ML Robustness',  icon: '<path d="M2 12h20"/><path d="M12 2v20"/><path d="M4.93 4.93l14.14 14.14"/><path d="M19.07 4.93 4.93 19.07"/>' },
                { href: '/observability', label: 'Observability',  icon: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>' },
                { href: '/news',          label: 'Новости',        icon: '<path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><path d="M18 14h-8"/><path d="M15 18h-5"/><path d="M10 6h8v4h-8V6Z"/>' },
                { href: '/trades',        label: 'История сделок', icon: '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>' },
                { href: '/logs',          label: 'Логи',           icon: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="16" y2="17"/>' }
            ]
        },
        {
            section: 'Система',
            items: [
                { href: '/settings',      label: 'Настройки',      icon: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>' }
            ]
        }
    ];

    function isActive(href, path) {
        if (href === '/') return path === '/' || path === '';
        return path === href || path.indexOf(href + '/') === 0;
    }

    function buildLink(item, active) {
        return '<a href="' + item.href + '" class="sidebar-link' + (active ? ' active' : '') + '">'
            + SVG_OPEN + item.icon + SVG_CLOSE
            + '<span>' + item.label + '</span>'
            + '</a>';
    }

    function renderSidebar() {
        var nav = document.getElementById('sidebar');
        if (!nav) return;
        // Idempotent: if already filled (server-rendered or earlier call), skip.
        if (nav.dataset.rendered === '1' || nav.children.length > 0) return;

        var path = window.location.pathname.replace(/\/+$/, '') || '/';
        var html = '';
        NAV_GROUPS.forEach(function (grp, i) {
            if (i > 0) {
                html += '<div class="sidebar-divider"></div>';
                if (grp.section) html += '<div class="sidebar-section">' + grp.section + '</div>';
            }
            grp.items.forEach(function (item) {
                html += buildLink(item, isActive(item.href, path));
            });
        });
        nav.innerHTML = html;
        nav.dataset.rendered = '1';
    }

    function initMobileSidebar() {
        var toggle = document.getElementById('mobile-menu-toggle');
        var sidebar = document.getElementById('sidebar');
        var backdrop = document.getElementById('mobile-sidebar-backdrop');
        if (!toggle || !sidebar || !backdrop) return;
        if (toggle.dataset.wired === '1') return;
        toggle.dataset.wired = '1';

        function openMenu() {
            sidebar.classList.add('open');
            backdrop.classList.add('active');
            toggle.setAttribute('aria-expanded', 'true');
            var firstLink = sidebar.querySelector('.sidebar-link');
            if (firstLink) firstLink.focus();
        }
        function closeMenu() {
            sidebar.classList.remove('open');
            backdrop.classList.remove('active');
            toggle.setAttribute('aria-expanded', 'false');
        }
        toggle.addEventListener('click', function () {
            sidebar.classList.contains('open') ? closeMenu() : openMenu();
        });
        backdrop.addEventListener('click', closeMenu);
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape' && sidebar.classList.contains('open')) closeMenu();
        });
        sidebar.querySelectorAll('.sidebar-link').forEach(function (link) {
            link.addEventListener('click', closeMenu);
        });
    }

    function init() {
        renderSidebar();
        initMobileSidebar();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
