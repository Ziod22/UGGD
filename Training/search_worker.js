// Long-running Mastodon search worker for GRPO reward. Reads one JSON request per line on stdin
// ({"id": <any>, "query": "<text>"}) and writes one JSON reply per line on stdout
// ({"id": <same>, "results": [{"url", "text"}, ...]} or {"id": <same>, "error": "<message>"}).
// Launches a single browser and reuses the saved login session's access token, so each query
// costs one API call instead of a full browser start. Searches run sequentially with a random
// 1-2 second pause after each one.
// Usage: node scripts/search_worker.js
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { chromium } = require('@playwright/test');

const CONFIG = {
  instance: 'mastodon.social',
  topN: 10,
};

const INSTANCE = process.env.MASTODON_INSTANCE || CONFIG.instance;
const TOP_N = Number(process.env.SEARCH_TOP_N) || CONFIG.topN;
const REPO_ROOT = path.join(__dirname, '..');
const SESSION_FILE = path.join(REPO_ROOT, 'tmp', 'auth', `mastodon_${INSTANCE}.json`);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function stripHtml(html) {
  if (!html) return '';
  return html
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/p>/gi, '\n\n')
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .trim();
}

async function extractToken(context) {
  const page = await context.newPage();
  try {
    await page.goto(`https://${INSTANCE}/home`, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#initial-state', { timeout: 15000 }).catch(() => {});
    return await page.evaluate(() => {
      const el = document.getElementById('initial-state');
      if (!el) return null;
      try {
        return JSON.parse(el.textContent)?.meta?.access_token || null;
      } catch {
        return null;
      }
    });
  } finally {
    await page.close();
  }
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ storageState: JSON.parse(fs.readFileSync(SESSION_FILE, 'utf8')) });
  let token = await extractToken(context);
  if (!token) {
    process.stderr.write(`no access token from ${INSTANCE}; session may have expired\n`);
    await browser.close();
    process.exit(1);
  }

  const rl = readline.createInterface({ input: process.stdin });
  for await (const line of rl) {
    if (!line.trim()) continue;
    const req = JSON.parse(line);
    let reply;
    try {
      const url = new URL(`https://${INSTANCE}/api/v2/search`);
      url.searchParams.set('q', req.query);
      url.searchParams.set('type', 'statuses');
      url.searchParams.set('limit', String(TOP_N));
      const res = await context.request.get(url.toString(), {
        headers: { Accept: 'application/json', Authorization: `Bearer ${token}` },
      });
      if (res.status() === 401) {
        token = await extractToken(context);
        reply = { id: req.id, error: 'unauthorized; token refreshed, retry' };
      } else if (!res.ok()) {
        reply = { id: req.id, error: `search returned ${res.status()}` };
      } else {
        const body = await res.json();
        const results = (body.statuses || []).slice(0, TOP_N).map((s) => ({
          url: s.url || s.uri || null,
          text: stripHtml(s.content),
        }));
        reply = { id: req.id, results };
      }
    } catch (err) {
      reply = { id: req.id, error: err.message };
    }
    process.stdout.write(JSON.stringify(reply) + '\n');
    await sleep(1000 + Math.random() * 1000);
  }
  await browser.close();
}

main().catch((err) => {
  process.stderr.write(err.stack + '\n');
  process.exit(1);
});
