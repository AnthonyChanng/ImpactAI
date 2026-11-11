import { defineConfig, devices } from "@playwright/test";
const live = process.env.E2E_DATASET === "1";
export default defineConfig({
  testDir: "./tests", fullyParallel: false, retries: 0, workers: 1, timeout: 60000,
  expect: {timeout: 10000},
  use: {baseURL: process.env.TEST_BASE_URL || (live ? "http://127.0.0.1:3000" : "http://127.0.0.1:3100"), trace: "retain-on-failure", screenshot: "only-on-failure"},
  webServer: live ? undefined : [
    {command: "node tests/fixture-server.mjs", url: "http://127.0.0.1:18001/health", reuseExistingServer: false},
    {command: "npm run dev -- --port 3100", url: "http://127.0.0.1:3100", timeout: 120000, reuseExistingServer: false, env: {BACKEND_URL: "http://127.0.0.1:18001", NEXT_TELEMETRY_DISABLED: "1"}},
  ],
  projects: [{name: "desktop", use: {...devices["Desktop Chrome"], channel: process.env.PLAYWRIGHT_CHANNEL}},
    {name: "mobile", use: {...devices["iPhone 13"], defaultBrowserType: "chromium", channel: process.env.PLAYWRIGHT_CHANNEL}}],
  reporter: [["list"], ["html", {open: "never"}]],
});
