// Fetches a single public Mastodon post (toot) given its URL, including its replies
// (author handle, text, timestamp) and everyone who favourited it.
//
// Usage: node fetch.js <post-url> [--output-dir <dir>]
//
// Output is printed to stdout and also saved as JSON under <output-dir>/ (default: ./output),
// as <instance>_<postId>.json -- this is the input format preprocess.py expects.
const fs = require('fs');
const path = require('path');

// Safety cap on paginated fetches
const MAX_PAGES = 25;

function parseArgs(argv) {
  const positional = [];
  let outputDir = path.join(__dirname, 'output');
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--output-dir') {
      outputDir = argv[++i];
    } else {
      positional.push(argv[i]);
    }
  }
  return { url: positional[0], outputDir };
}

function parsePostUrl(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch {
    console.error(`Not a valid URL: ${raw}`);
    process.exit(1);
  }
  const segments = url.pathname.split('/').filter(Boolean);
  const id = [...segments].reverse().find((s) => /^\d+$/.test(s));
  if (!id) {
    console.error(`Could not find a numeric status ID in the URL path: ${url.pathname}`);
    process.exit(1);
  }
  return { instance: url.hostname, id };
}

function stripHtml(html) {
  if (!html) return null;
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

async function getJson(url) {
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    console.error(`Request failed: ${res.status} ${res.statusText} (${url})`);
    process.exit(1);
  }
  return res.json();
}

function nextPageUrl(linkHeader) {
  if (!linkHeader) return null;
  const next = linkHeader.split(',').find((part) => part.includes('rel="next"'));
  return next ? next.trim().match(/^<(.+)>/)?.[1] ?? null : null;
}

async function getAllPages(url) {
  let results = [];
  let pageUrl = url;
  let pages = 0;
  let truncated = false;
  while (pageUrl && pages < MAX_PAGES) {
    const res = await fetch(pageUrl, { headers: { Accept: 'application/json' } });
    if (!res.ok) {
      console.error(`Request failed: ${res.status} ${res.statusText} (${pageUrl})`);
      process.exit(1);
    }
    results = results.concat(await res.json());
    pageUrl = nextPageUrl(res.headers.get('link'));
    pages += 1;
  }
  if (pageUrl) truncated = true;
  return { results, truncated };
}

async function fetchPost(input, outputDir) {
  const { instance, id } = parsePostUrl(input);
  const status = await getJson(`https://${instance}/api/v1/statuses/${id}`);
  const original = status.reblog || status;
  const context = await getJson(`https://${instance}/api/v1/statuses/${original.id}/context`);
  const replies = context.descendants.map((r) => ({
    id: r.id,
    authorId: r.account?.id || null,
    handle: r.account?.acct ? `@${r.account.acct}` : null,
    text: stripHtml(r.content),
    createdAt: r.created_at || null,
    inReplyToId: r.in_reply_to_id || null,
  }));
  // favourited by with limit set to 80
  const { results: favouritedByAccounts, truncated: favouritesTruncated } = await getAllPages(
    `https://${instance}/api/v1/statuses/${original.id}/favourited_by?limit=80`
  );
  const favouritedBy = favouritedByAccounts.map((a) => ({
    userId: a.id,
    handle: a.acct ? `@${a.acct}` : null,
    displayName: a.display_name || null,
    url: a.url || null,
    followersCount: a.followers_count ?? null,
    bot: a.bot ?? null,
  }));
  if (favouritesTruncated) {
    console.error(`Warning: favouritedBy truncated at ${favouritedBy.length} accounts (hit the ${MAX_PAGES}-page fetch cap).`);
  }
  const post = {
    id: original.id,
    authorId: status.account?.id || null,
    author: status.account?.display_name || null,
    handle: status.account?.acct ? `@${status.account.acct}` : null,
    boostOf: status.reblog ? (status.reblog.account?.acct ? `@${status.reblog.account.acct}` : null) : null,
    text: stripHtml(original.content),
    spoilerText: original.spoiler_text || null,
    createdAt: status.created_at || null,
    url: status.url || status.uri || null,
    language: original.language || null,
    visibility: status.visibility || null,
    stats: {
      reply: original.replies_count ?? null,
      reblog: original.reblogs_count ?? null,
      favourite: original.favourites_count ?? null,
    },
    mediaAttachments: (original.media_attachments || []).map((m) => ({
      type: m.type,
      url: m.url,
      description: m.description,
    })),
    tags: (original.tags || []).map((t) => t.name),
    replies,
    favouritedBy,
  };
  fs.mkdirSync(outputDir, { recursive: true });
  const outFile = path.join(outputDir, `${instance}_${original.id}.json`);
  fs.writeFileSync(outFile, JSON.stringify(post, null, 2));
  return { post, outFile };
}

if (require.main === module) {
  const { url, outputDir } = parseArgs(process.argv.slice(2));
  if (!url) {
    console.error('Usage: node fetch.js <mastodon-post-url> [--output-dir <dir>]');
    process.exit(1);
  }
  fetchPost(url, outputDir).then(({ post, outFile }) => {
    console.log(JSON.stringify(post, null, 2));
    console.error(`Saved output to ${outFile}`);
  });
}

module.exports = { fetchPost, parsePostUrl, stripHtml };
