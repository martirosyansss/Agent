const { defineConfig } = require('vitest/config');
const path = require('node:path');

module.exports = defineConfig({
    test: {
        environment: 'node',
        include: ['tests/**/*.test.js'],
        coverage: {
            reporter: ['text', 'html'],
            include: ['../static/js/lib/**/*.js'],
        },
    },
    resolve: {
        alias: {
            '@lib': path.resolve(__dirname, '../static/js/lib'),
        },
    },
});
