'use strict';
/**
 * Adds the Lower League Predictor's Betfair football recorder to the
 * Formintel odds proxy. Run it once from the proxy's folder (the one with
 * server.js), after downloading football.js and this file into it:
 *
 *     cd /root/formintel-proxy
 *     curl -fsSLO https://raw.githubusercontent.com/paulhessle-cyber/Football-formintel/main/server/football.js
 *     curl -fsSLO https://raw.githubusercontent.com/paulhessle-cyber/Football-formintel/main/server/install-football.js
 *     node install-football.js
 *
 * It backs up server.js first, checks the edited file before saving it, and
 * does nothing if the recorder is already in. Then restart the proxy.
 * To update football.js later, download it again and restart - no reinstall.
 */
const fs = require('fs');
const path = require('path');
const cp = require('child_process');

const dir = __dirname;
const serverFile = path.join(dir, 'server.js');
const modFile = path.join(dir, 'football.js');

function say(s) { console.log(s); }
function stop(s) { console.error('✗ ' + s); process.exit(1); }

if (!fs.existsSync(serverFile)) stop('No server.js here (' + dir + '). Put football.js and this file in the proxy folder and run it from there.');
if (!fs.existsSync(modFile)) stop('football.js is missing — download it into ' + dir + ' first.');

// The proxy's .env (FI_FOOTBALL_DIR, PORT), as server.js reads it
try {
  fs.readFileSync(path.join(dir, '.env'), 'utf8').split('\n').forEach((line) => {
    const t = line.trim(), eq = t.indexOf('=');
    if (!t || t[0] === '#' || eq === -1) return;
    const k = t.slice(0, eq).trim();
    const v = t.slice(eq + 1).trim().replace(/^(['"])(.*)\1$/, '$2');
    if (!(k in process.env)) process.env[k] = v;
  });
} catch (e) {}

// 1. Where the prices will be saved
try {
  const d = require(modFile).diagnose();
  if (!d.writable) stop('Can\'t write to ' + d.dir + '. Add FI_FOOTBALL_DIR=/a/folder/it/can/write to .env, or fix that folder\'s permissions, then run this again.');
  say('✓ Prices will be saved in ' + d.dir + (d.days ? ' (' + d.days + ' day' + (d.days === 1 ? '' : 's') + ' there already)' : ''));
} catch (e) { stop('football.js failed to load: ' + e.message); }

// 2. Patch server.js
let s = fs.readFileSync(serverFile, 'utf8');
const hasReq = /require\(['"]\.\/football(\.js)?['"]\)/.test(s);
const hasRoute = /path === ['"]\/football['"]/.test(s);
const hasStart = /fiFootball\.start\(/.test(s);

if (hasReq && hasRoute && hasStart) {
  say('✓ server.js already has the football recorder — nothing to change.');
} else {
  if (!/function\s+betfairRpc\s*\(/.test(s)) {
    stop('Couldn\'t find betfairRpc() in server.js — the recorder uses it to share the proxy\'s Betfair login. Nothing was changed.');
  }

  if (!hasReq) {
    const lines = s.split('\n');
    let at = -1;
    for (let i = 0; i < Math.min(lines.length, 150); i++) if (/^\s*const\s+\w+\s*=\s*require\(/.test(lines[i])) at = i;
    if (at === -1) stop('Couldn\'t find the require lines at the top of server.js. Nothing was changed.');
    lines.splice(at + 1, 0, "const fiFootball = require('./football');   // Lower League Predictor: Betfair football prices (/football)");
    s = lines.join('\n');
  }

  if (!hasRoute) {
    const route =
      "    // GET /football?days=14 — Betfair prices for League One, League Two and National League matches\n" +
      "    if (path === '/football') {\n" +
      "      const out = await fiFootball.handle(q, req.headers);\n" +
      "      Object.keys(out.headers).forEach((h) => res.setHeader(h, out.headers[h]));\n" +
      "      res.writeHead(out.status);\n" +
      "      res.end(out.body);\n" +
      "      return;\n" +
      "    }\n\n";
    const anchors = [/^([ \t]*)if \(path === ['"]\/late['"]\)/m, /^([ \t]*)if \(path === ['"]\/meetings['"]\)/m,
                     /^([ \t]*)if \(path === ['"]\/odds['"]\)/m, /^([ \t]*)if \(path === ['"]\/health['"]\)/m];
    let done = false;
    for (let i = 0; i < anchors.length && !done; i++) {
      const m = s.match(anchors[i]);
      if (m) { s = s.slice(0, m.index) + route + s.slice(m.index); done = true; }
    }
    if (!done) stop('Couldn\'t find where the routes are in server.js (looked for /late, /meetings, /odds, /health). Nothing was changed.');
  }

  if (!hasStart) {
    const startLines =
      "// Lower League Predictor: Betfair football prices for League One, League Two and National League\n" +
      "fiFootball.start({ rpc: betfairRpc });\n\n";
    const m = s.match(/^server\.listen\(/m);
    if (m) s = s.slice(0, m.index) + startLines + s.slice(m.index);
    else s = s.replace(/\s*$/, '\n\n') + startLines;
  }

  const backup = serverFile + '.before-football';
  const tmp = path.join(dir, '.server.football-check.js');
  fs.writeFileSync(tmp, s);
  try { cp.execFileSync(process.execPath, ['--check', tmp], { stdio: 'pipe' }); }
  catch (e) { try { fs.unlinkSync(tmp); } catch (x) {} stop('The edited server.js wouldn\'t run, so it was NOT saved:\n' + String(e.stderr || e.message)); }
  try { fs.unlinkSync(tmp); } catch (x) {}
  fs.copyFileSync(serverFile, backup);
  fs.writeFileSync(serverFile, s);
  say('✓ server.js updated (the old one is saved as ' + path.basename(backup) + ')');
}

// 3. How to restart
let unit = '';
try {
  const out = cp.execSync('systemctl list-units --type=service --all --no-legend --plain 2>/dev/null', { encoding: 'utf8' });
  const hit = out.split('\n').map((l) => l.trim().split(/\s+/)[0]).filter(Boolean)
    .filter((u) => /formintel|proxy|odds/i.test(u));
  if (hit.length) unit = hit[0];
} catch (e) {}
let pm2 = false;
if (!unit) { try { cp.execSync('pm2 jlist', { stdio: 'pipe' }); pm2 = true; } catch (e) {} }
say('\nNow restart the proxy:');
say(unit ? '   systemctl restart ' + unit
         : pm2 ? '   pm2 restart all' : '   (restart it the way you normally do — e.g. systemctl restart <service name>)');
say('Then check it (give it a minute):   curl -s "http://localhost:' + (process.env.PORT || '3001') + '/football?status=1"');
say('It should list three competitions (League 1, League 2, National League) and a number of markets followed.');
