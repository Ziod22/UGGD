// Prints the URL of the first Mastodon search result for a query, using the saved login session.
// Usage: node scripts/first-search-result.js <query> [instance]
// The query is required input (it's different on every call, not a run setting), so it stays a
// command-line argument. The default instance is a config value, editable below.
// Prints just the URL on stdout; exits 1 with a message on stderr if there are no results.
const fs = require('fs');
const path = require('path');
const { chromium } = require('@playwright/test');

const CONFIG = {
  instance: 'mastodon.social',
};

const query = process.argv[2];
const instance = process.argv[3] || CONFIG.instance;
if (!query) {
  console.error('Usage: node scripts/first-search-result.js <query> [instance]');
  process.exit(1);
}

const sessionFile = path.join(__dirname, '..', 'tmp', 'auth', `mastodon_${instance}.json`);
if (!fs.existsSync(sessionFile)) {
  console.error(`No saved session at ${sessionFile} — run \`npm run login:mastodon -- ${instance}\` first.`);
  process.exit(1);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({ storageState: JSON.parse(fs.readFileSync(sessionFile, 'utf8')) });
    const page = await context.newPage();
    await page.goto(`https://${instance}/home`, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#initial-state', { timeout: 15000 }).catch(() => {});
    const token = await page.evaluate(() => {
      const el = document.getElementById('initial-state');
      if (!el) return null;
      try {
        return JSON.parse(el.textContent)?.meta?.access_token || null;
      } catch {
        return null;
      }
    });
    if (!token) {
      console.error(`Couldn't get an access token from ${instance} — the session may have expired.`);
      process.exit(1);
    }

    const url = new URL(`https://${instance}/api/v2/search`);
    url.searchParams.set('q', query);
    url.searchParams.set('type', 'statuses');
    url.searchParams.set('limit', '1');
    const res = await context.request.get(url.toString(), {
      headers: { Accept: 'application/json', Authorization: `Bearer ${token}` },
    });
    if (!res.ok()) {
      console.error(`${instance} search returned ${res.status()} ${res.statusText()}`);
      process.exit(1);
    }

    const first = ((await res.json()).statuses || [])[0];
    console.log(first ? first.url || first.uri : 'None');
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
