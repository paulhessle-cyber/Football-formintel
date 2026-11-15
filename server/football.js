'use strict';
/**
 * Lower League Predictor: Betfair football price recorder, an add-on for the
 * Formintel odds proxy (the same proxy that serves FormIntel's horse racing
 * prices). It shares the proxy's own Betfair session, so it never logs in.
 *
 * For every English League One, League Two and National League match it
 * looks at the Betfair Exchange Match Odds and Over/Under 2.5 Goals markets:
 *   every 30 minutes from 3 days before kick-off,
 *   every 10 minutes in the last 6 hours,
 *   every 2 minutes in the last hour (team sheets come out about then),
 * and keeps the best back and lay price for each outcome at each look, plus
 * the money matched. The last look before the match goes in-play is the
 * Betfair closing price. Afterwards it records the result (who won the
 * Match Odds market and whether Over 2.5 won).
 *
 *   GET /football?days=14        matches kicking off in the last 14 days and the next 3
 *   GET /football?status=1       what it's doing and what Betfair is sending
 *
 * One file per kick-off day in football-data/ next to server.js (or
 * FI_FOOTBALL_DIR in .env). Works with a delayed app key.
 *
 * install-football.js adds it to server.js: a require line, the /football
 * route and   fiFootball.start({ rpc: betfairRpc });
 */
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const VERSION = 1;
// Betfair's competition names (matched loosely; North/South are excluded)
const LEAGUES = [
  { div: 'E2', re: /^english\s+league\s*(1|one)$/i },
  { div: 'E3', re: /^english\s+league\s*(2|two)$/i },
  { div: 'EC', re: /^english\s+national\s+league$/i },
];
const MARKET_TYPES = ['MATCH_ODDS', 'OVER_UNDER_25'];
const AHEAD_MS = 3 * 86400000;                       // follow matches from this long before kick-off
const TICK_MS = 30000;
const CATALOGUE_MS = 30 * 60000;
const COMPETITIONS_MS = 12 * 3600000;
const GIVE_UP_MS = 4 * 3600000;                      // a match not in-play 4h after kick-off is dropped
const RESULT_FOR_MS = 8 * 3600000;                   // keep asking for the result this long after kick-off
const MAX_LOOKS = 260;
const KEEP_DAYS = 800;
const MAX_DAYS_SERVED = 400;

function gapFor(msToKo) {                            // how often to look, by time to kick-off
  if (msToKo <= 60 * 60000) return 2 * 60000;
  if (msToKo <= 6 * 3600000) return 10 * 60000;
  return 30 * 60000;
}

let rpc = null, dir = '', started = false, timer = null, kick = null, busy = false;
let now = () => Date.now();
let maxPerCall = 30;                                 // EX_BEST_OFFERS depth 1, weight 5 per market: 30 x 5 = 150, under 200
let compIds = {}, lastComps = 0, lastCatalogue = 0, backoffUntil = 0, failsInRow = 0, lastPrune = 0, lastWarn = 0;
const mk = new Map();                                // marketId -> market being followed
const days = {};                                     // 'YYYY-MM-DD' -> { v, date, matches: { eventId: match } }
const dirty = new Set();
let saveTimer = null, respCache = null;
const stat = { since: 0, ticks: 0, calls: 0, errors: 0, lastError: '', lastErrorAt: 0, delayed: null,
               competitions: {}, catalogueAt: 0, markets: 0 };

// ── helpers ──────────────────────────────────────────────────────────────────
function num(v) {
  if (typeof v === 'number') return isFinite(v) ? v : null;
  if (typeof v === 'string' && v.trim() !== '' && isFinite(+v)) return +v;
  return null;
}
function r2(x) { return x == null ? null : Math.round(x * 100) / 100; }
function ukDate(ms) { return new Date(ms).toLocaleDateString('en-CA', { timeZone: 'Europe/London' }); }
function ukTime(ms) { return new Date(ms).toLocaleTimeString('en-GB', { timeZone: 'Europe/London', hour: '2-digit', minute: '2-digit', hour12: false }); }
function addDays(iso, n) { const d = new Date(iso + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() + n); return d.toISOString().slice(0, 10); }
function chunks(a, n) { const out = []; for (let i = 0; i < a.length; i += n) out.push(a.slice(i, i + n)); return out; }
function warn(msg) {
  const t = now();
  if (t - lastWarn < 60000) return;
  lastWarn = t;
  console.warn('football: ' + msg);
}
function noteError(e) {
  stat.errors++; stat.lastError = String((e && e.message) || e).slice(0, 300); stat.lastErrorAt = now();
  warn(stat.lastError);
}
function dataDir() { return dir || process.env.FI_FOOTBALL_DIR || path.join(__dirname, 'football-data'); }
function dayFile(date) { return path.join(dataDir(), date + '.json'); }

async function call(method, params) {
  stat.calls++;
  try {
    const res = await rpc(method, params);
    failsInRow = 0;
    return res;
  } catch (e) {
    failsInRow++;
    const msg = String(e && e.message);
    if (/TOO_MUCH_DATA/.test(msg) && maxPerCall > 1) maxPerCall = Math.max(1, Math.floor(maxPerCall / 2));
    // Back off rather than keep knocking: a failed call can mean a fresh login
    // attempt by the proxy, and repeated failed logins can lock the account.
    if (/login failed|INVALID_USERNAME|ACCOUNT|BANNED|TOO_MANY/i.test(msg)) backoffUntil = now() + 15 * 60000;
    else if (failsInRow >= 5) backoffUntil = now() + Math.min(30 * 60000, 60000 * Math.pow(2, Math.min(5, failsInRow - 5)));
    throw e;
  }
}

// ── day files ────────────────────────────────────────────────────────────────
function readDayFile(date) {
  try {
    const f = dayFile(date);
    if (!fs.existsSync(f)) return null;
    const j = JSON.parse(fs.readFileSync(f, 'utf8'));
    return j && j.matches ? j : null;
  } catch (e) { noteError(new Error('could not read ' + date + '.json: ' + e.message)); return null; }
}
function day(date) {
  if (!days[date]) days[date] = readDayFile(date) || { v: VERSION, date: date, matches: {} };
  return days[date];
}
function saveSoon(date) {
  dirty.add(date);
  respCache = null;
  if (saveTimer) return;
  saveTimer = setTimeout(flush, 3000);
  if (saveTimer.unref) saveTimer.unref();
}
function flush() {
  saveTimer = null;
  try { fs.mkdirSync(dataDir(), { recursive: true }); } catch (e) {}
  Array.from(dirty).forEach((date) => {
    dirty.delete(date);
    const d = days[date];
    if (!d) return;
    try {
      const f = dayFile(date), tmp = f + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify(d));
      fs.renameSync(tmp, f);
    } catch (e) { noteError(new Error('could not save ' + date + '.json: ' + e.message)); }
  });
}

// ── which matches to follow ──────────────────────────────────────────────────
async function refreshCompetitions(t) {
  const list = await call('listCompetitions', { filter: { eventTypeIds: ['1'], marketCountries: ['GB'] } });
  const found = {};
  (list || []).forEach((c) => {
    const name = String((c.competition && c.competition.name) || '').trim();
    const id = c.competition && c.competition.id;
    LEAGUES.forEach((L) => { if (id && L.re.test(name)) found[L.div] = { id: String(id), name: name }; });
  });
  if (Object.keys(found).length) { compIds = found; stat.competitions = found; }
  lastComps = t;
}

function matchFor(ev, div, sch) {
  const date = ukDate(sch);
  const d = day(date);
  let m = d.matches[ev.id];
  if (!m) {
    const name = String(ev.name || '');
    const parts = name.split(/\s+v(?:s)?\.?\s+/i);
    m = d.matches[ev.id] = { id: String(ev.id), div: div, d: date, t: ukTime(sch), ko: sch,
                             home: (parts[0] || name).trim(), away: (parts[1] || '').trim(), mk: {} };
    saveSoon(date);
  }
  if (m.ko !== sch) { m.ko = sch; m.t = ukTime(sch); saveSoon(date); }
  return m;
}

async function refreshCatalogue(t) {
  const ids = Object.keys(compIds).map((k) => compIds[k].id);
  if (!ids.length) return;
  const byComp = {};
  Object.keys(compIds).forEach((k) => { byComp[compIds[k].id] = k; });
  const list = await call('listMarketCatalogue', {
    filter: { eventTypeIds: ['1'], competitionIds: ids, marketTypeCodes: MARKET_TYPES,
              marketStartTime: { from: new Date(t - 3 * 3600000).toISOString(), to: new Date(t + AHEAD_MS).toISOString() } },
    marketProjection: ['EVENT', 'COMPETITION', 'MARKET_START_TIME', 'MARKET_DESCRIPTION', 'RUNNER_DESCRIPTION'],
    sort: 'FIRST_TO_START',
    maxResults: 1000
  });
  lastCatalogue = t; stat.catalogueAt = t;
  (list || []).forEach((c) => {
    const sch = Date.parse(c.marketStartTime);
    const div = byComp[String(c.competition && c.competition.id)];
    const type = (c.description && c.description.marketType) || (/over\/under 2\.5/i.test(c.marketName || '') ? 'OVER_UNDER_25' : 'MATCH_ODDS');
    if (!c.marketId || !c.event || isNaN(sch) || !div || MARKET_TYPES.indexOf(type) === -1) return;
    const m = matchFor(c.event, div, sch);
    if (m.mk[type] && m.mk[type].id !== c.marketId) return;
    if (!m.mk[type]) {
      // runner order: Match Odds -> home, away, draw (by sortPriority); O/U -> over, under
      const runners = (c.runners || []).slice().sort((a, b) => (a.sortPriority || 0) - (b.sortPriority || 0));
      let sel;
      if (type === 'MATCH_ODDS') {
        const draw = runners.find((r) => /^the draw$/i.test(r.runnerName || ''));
        const teams = runners.filter((r) => r !== draw);
        sel = { h: teams[0] && teams[0].selectionId, d: draw && draw.selectionId, a: teams[1] && teams[1].selectionId };
        if (teams[0] && teams[0].runnerName) m.home = teams[0].runnerName;
        if (teams[1] && teams[1].runnerName) m.away = teams[1].runnerName;
      } else {
        const over = runners.find((r) => /^over/i.test(r.runnerName || ''));
        const under = runners.find((r) => /^under/i.test(r.runnerName || ''));
        sel = { o: over && over.selectionId, u: under && under.selectionId };
      }
      m.mk[type] = { id: c.marketId, sel: sel, looks: [] };
      saveSoon(m.d);
    }
    if (!mk.has(c.marketId) && !m.mk[type].done) mk.set(c.marketId, { id: c.marketId, type: type, match: m, lastLook: 0, ph: 'pre' });
  });
  stat.markets = mk.size;
}

// ── looking at prices ────────────────────────────────────────────────────────
function due(M, t) {
  const m = M.match;
  if (M.ph === 'pre') {
    if (t > m.ko + GIVE_UP_MS) { mk.delete(M.id); return false; }
    return !M.lastLook || t - M.lastLook >= gapFor(m.ko - t);
  }
  if (M.ph === 'live') {                              // waiting for the result
    if (t > m.ko + RESULT_FOR_MS) { mk.delete(M.id); return false; }
    return t - M.lastLook >= 10 * 60000;
  }
  return false;
}

async function look(t) {
  const list = [];
  mk.forEach((M) => { if (due(M, t)) list.push(M); });
  if (!list.length) return;
  for (const group of chunks(list, maxPerCall)) {
    const books = await call('listMarketBook', {
      marketIds: group.map((M) => M.id),
      priceProjection: { priceData: ['EX_BEST_OFFERS'], exBestOffersOverrides: { bestPricesDepth: 1 }, virtualise: true }
    });
    const at = now(), byId = {};
    (books || []).forEach((b) => { if (b && b.marketId) byId[b.marketId] = b; });
    group.forEach((M) => { M.lastLook = at; if (byId[M.id]) onBook(M, byId[M.id], at); });
  }
}

function price(r) {
  const back = r && r.ex && r.ex.availableToBack && r.ex.availableToBack[0];
  const lay = r && r.ex && r.ex.availableToLay && r.ex.availableToLay[0];
  return [back ? r2(num(back.price)) : null, lay ? r2(num(lay.price)) : null];
}

function onBook(M, b, at) {
  if (typeof b.isMarketDataDelayed === 'boolean') stat.delayed = b.isMarketDataDelayed;
  const m = M.match, rec = m.mk[M.type];
  if (M.ph === 'pre') {
    if (b.inplay || b.status === 'CLOSED') {
      M.ph = 'live';
      rec.inplayAt = at;
      if (b.status === 'CLOSED') settle(M, b);
      saveSoon(m.d);
      return;
    }
    if (b.status !== 'OPEN') return;                  // suspended: no usable prices
    const byId = {};
    (b.runners || []).forEach((r) => { byId[r.selectionId] = r; });
    const keys = Object.keys(rec.sel);
    const look = { at: at, mt: Math.round(num(b.totalMatched) || 0) };
    keys.forEach((k) => { look[k] = price(byId[rec.sel[k]]); });
    rec.looks.push(look);
    if (rec.looks.length > MAX_LOOKS) rec.looks.splice(1, 1);   // keep the first look, drop the oldest after it
    saveSoon(m.d);
    return;
  }
  if (M.ph === 'live' && b.status === 'CLOSED') { settle(M, b); saveSoon(m.d); }
}

function settle(M, b) {
  const rec = M.match.mk[M.type];
  const winner = (b.runners || []).find((r) => r.status === 'WINNER');
  if (winner) {
    const k = Object.keys(rec.sel).find((x) => rec.sel[x] === winner.selectionId);
    rec.won = k || null;
  } else rec.won = 'void';
  rec.done = 1;
  M.ph = 'done';
  mk.delete(M.id);
}

// ── housekeeping ─────────────────────────────────────────────────────────────
function housekeeping(t) {
  const today = ukDate(t), keep = {};
  for (let i = -2; i <= 4; i++) keep[addDays(today, i)] = 1;
  Object.keys(days).forEach((d) => { if (!keep[d] && !dirty.has(d)) delete days[d]; });
  if (t - lastPrune < 6 * 3600000) return;
  lastPrune = t;
  try {
    const cutoff = addDays(today, -KEEP_DAYS);
    fs.readdirSync(dataDir()).forEach((f) => {
      const mm = /^(\d{4}-\d{2}-\d{2})\.json$/.exec(f);
      if (mm && mm[1] < cutoff) fs.unlinkSync(path.join(dataDir(), f));
    });
  } catch (e) {}
}

async function tick() {
  if (busy || !rpc || !started) return;
  busy = true;
  stat.ticks++;
  try {
    const t = now();
    if (t < backoffUntil) return;
    if (!lastComps || t - lastComps >= COMPETITIONS_MS || !Object.keys(compIds).length && t - lastComps >= 30 * 60000) {
      try { await refreshCompetitions(t); } catch (e) { noteError(e); }
    }
    if (!lastCatalogue || t - lastCatalogue >= CATALOGUE_MS) {
      try { await refreshCatalogue(t); } catch (e) { noteError(e); }
    }
    try { await look(now()); } catch (e) { noteError(e); }
    housekeeping(now());
  } catch (e) { noteError(e); }
  finally { busy = false; }
}

// After a restart: pick up recent matches still waiting for a result.
function resume() {
  const today = ukDate(now());
  [addDays(today, -1), today].forEach((date) => {
    const d = day(date);
    Object.keys(d.matches).forEach((id) => {
      const m = d.matches[id];
      Object.keys(m.mk).forEach((type) => {
        const rec = m.mk[type];
        if (!rec.done && rec.inplayAt && !mk.has(rec.id)) mk.set(rec.id, { id: rec.id, type: type, match: m, lastLook: 0, ph: 'live' });
      });
    });
  });
}

// ── public ───────────────────────────────────────────────────────────────────
function start(opts) {
  opts = opts || {};
  if (typeof opts.rpc !== 'function') { console.warn('football: start() needs the proxy\'s betfairRpc — not recording.'); return; }
  rpc = opts.rpc;
  if (opts.dir) dir = opts.dir;
  if (typeof opts.now === 'function') now = opts.now;
  if (started) return;
  started = true;
  stat.since = now();
  try { fs.mkdirSync(dataDir(), { recursive: true }); } catch (e) { noteError(e); }
  resume();
  if (opts.timers === false) return;
  kick = setTimeout(tick, 20000);
  timer = setInterval(tick, TICK_MS);
  console.log('Football price recorder on: saving to ' + dataDir());
}

function stop() {
  if (timer) clearInterval(timer);
  if (kick) clearTimeout(kick);
  timer = null; kick = null; started = false;
  if (saveTimer) { clearTimeout(saveTimer); flush(); }
}

function fileDates() {
  try { return fs.readdirSync(dataDir()).map((f) => (/^(\d{4}-\d{2}-\d{2})\.json$/.exec(f) || [])[1]).filter(Boolean).sort(); }
  catch (e) { return []; }
}

function status() {
  const t = now(), files = fileDates();
  let following = 0, waitingResult = 0;
  mk.forEach((M) => { if (M.ph === 'pre') following++; else if (M.ph === 'live') waitingResult++; });
  return {
    ok: !!started, version: VERSION, since: stat.since ? new Date(stat.since).toISOString() : null,
    competitions: stat.competitions, firstDay: files[0] || null, days: files.length,
    marketsFollowed: following, awaitingResult: waitingResult,
    catalogueAt: stat.catalogueAt ? new Date(stat.catalogueAt).toISOString() : null,
    delayedData: stat.delayed, betfairCalls: stat.calls, errors: stat.errors,
    lastError: stat.lastError || null, lastErrorAt: stat.lastErrorAt ? new Date(stat.lastErrorAt).toISOString() : null,
    now: new Date(t).toISOString()
  };
}

function send(body, headers, code) {
  const out = { status: code || 200, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }, body: body };
  const ae = String((headers && headers['accept-encoding']) || '');
  if (body.length > 1024 && /\bgzip\b/.test(ae)) {
    out.body = zlib.gzipSync(body);
    out.headers['Content-Encoding'] = 'gzip';
    out.headers['Vary'] = 'Accept-Encoding';
  }
  return out;
}

// For the route in server.js: always resolves, never throws.
async function handle(q, headers) {
  try {
    q = q || {};
    const one = (v) => (Array.isArray(v) ? v[0] : v);
    if (one(q.status)) return send(JSON.stringify(status()), headers);
    const today = ukDate(now());
    const n = Math.max(1, Math.min(MAX_DAYS_SERVED, parseInt(one(q.days), 10) || 14));
    const first = addDays(today, -(n - 1)), last = addDays(today, 3);
    const dates = fileDates().filter((d) => d >= first && d <= last);
    const key = dates.join(',') + '|' + /\bgzip\b/.test(String((headers && headers['accept-encoding']) || ''));
    if (respCache && respCache.key === key && now() - respCache.at < 60000) return respCache.out;
    const matches = [];
    dates.forEach((d) => {
      const f = days[d] || readDayFile(d);
      if (f) Object.keys(f.matches).forEach((id) => matches.push(f.matches[id]));
    });
    matches.sort((a, b) => (a.ko || 0) - (b.ko || 0));
    const body = JSON.stringify({ v: VERSION, from: first, to: last, status: status(), matches: matches });
    const out = send(body, headers);
    respCache = { key: key, at: now(), out: out };
    return out;
  } catch (e) {
    noteError(e);
    return send(JSON.stringify({ error: 'football: ' + e.message }), headers, 500);
  }
}

function diagnose() {
  const d = dataDir();
  let writable = false;
  try { fs.mkdirSync(d, { recursive: true }); const f = path.join(d, '.write-test'); fs.writeFileSync(f, 'ok'); fs.unlinkSync(f); writable = true; } catch (e) {}
  const files = fileDates();
  return { dir: d, writable: writable, days: files.length, firstDay: files[0] || null, lastDay: files[files.length - 1] || null };
}

module.exports = { start, stop, handle, status, diagnose, _tick: tick, _flush: flush, _state: () => ({ mk, days, stat, compIds }) };
