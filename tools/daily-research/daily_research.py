#!/usr/bin/env python3
"""
daily_research.py — 每天两次自动跑：GitHub / Reddit 搜新项目，与本地项目对比，写研报，推送到 ai-manga-factory。

用法：
  python tools/daily-research/daily_research.py             # 跑一次
  python tools/daily-research/daily_research.py --dry-run  # 只读，不写不推

退出码：
  0 = 跑完（有/无发现）
  1 = 致命错误（credential / 配置）
"""
import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib import request, parse, error

WORK = Path(r'F:\new ai factory\tools\daily-research')
STATE = WORK / 'last-state.json'
OUTPUTS = WORK / 'outputs'
RUNS = WORK / 'RUNS.md'
CONFIG = WORK / 'config.json'
GIT_TOKEN_HELPER_CMD = ['git', 'credential-manager', 'get']
GIT_TOKEN_INPUT = 'protocol=https\nhost=github.com'
REPO = 'DeanChen85/ai-manga-factory'
REPO_DIR = Path(r'C:\Users\Dean\AppData\Local\Temp\dsh-ai-manga-factory-research')


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', file=sys.stderr, flush=True)


def get_github_token():
    """Read the GitHub credential from Windows Credential Manager via git credential-manager."""
    try:
        out = subprocess.run(
            GIT_TOKEN_HELPER_CMD,
            input=GIT_TOKEN_INPUT,
            capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError as e:
        log(f'credential-manager failed: {e.stderr}')
        sys.exit(1)
    for line in out.splitlines():
        if line.startswith('password='):
            return line[len('password='):].strip()
    log('no password found in credential-manager output')
    sys.exit(1)


def gh_get(url, token):
    req = request.Request(url, headers={
        'Authorization': f'token {token}',
        'User-Agent': 'dsh-agent-daily-research',
        'Accept': 'application/vnd.github+json',
    })
    try:
        with request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())
    except error.HTTPError as e:
        if e.code == 403:
            log(f'rate limited: {url}')
            return None
        raise


def gh_search_repos(token, query, days_back):
    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime('%Y-%m-%d')
    q = f'{query} created:>{since}'
    url = f'https://api.github.com/search/repositories?q={parse.quote(q)}&sort=stars&order=desc&per_page=15'
    data = gh_get(url, token)
    return data.get('items', []) if data else []


def reddit_rss(sub):
    # Reddit blocks most script requests; try with browser-like UA but fall back gracefully
    url = f'https://www.reddit.com/r/{sub}/new.json?limit=25'
    req = request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
    })
    try:
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log(f'reddit fetch failed for r/{sub}: {e}')
        return None

    class Entry:
        def __init__(self, post):
            self.id = post['name']
            self.title = post['title']
            self.url = 'https://reddit.com' + post['permalink']
            self.summary = (post.get('selftext') or '')[:200]

    class Root:
        def __init__(self, entries):
            self.entries = entries

    children = data.get('data', {}).get('children', [])
    return Root([Entry(c['data']) for c in children])


def hn_search(query):
    """Search Hacker News via the Algolia API. Free, no auth."""
    url = f'https://hn.algolia.com/api/v1/search?query={parse.quote(query)}&tags=story&numericFilters=created_at_i>{int((datetime.now(timezone.utc) - timedelta(days=14)).timestamp())}'
    req = request.Request(url, headers={'User-Agent': 'dsh-agent-daily-research'})
    try:
        with request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        log(f'HN fetch failed for query "{query}": {e}')
        return []

    class HnEntry:
        def __init__(self, hit):
            self.id = str(hit.get('objectID'))
            self.title = hit.get('title') or hit.get('story_text', '')[:120]
            self.url = hit.get('url') or f"https://news.ycombinator.com/item?id={self.id}"
            self.summary = (hit.get('story_text') or '')[:200]

    return [HnEntry(h) for h in data.get('hits', [])]


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'seenRepos': {}, 'seenPosts': {}, 'lastRun': None}


def save_state(state):
    state['lastRun'] = datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')


def load_config():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def compare_to_our_project(item):
    """
    Quick comparison: identify what the new project has that our pipeline lacks,
    based on the README's claims. Returns list of action items (strings), or [].
    """
    actions = []
    text = ' '.join(str(x) for x in [
        item.get('description') or '',
        item.get('name') or '',
    ]).lower()
    our_capabilities = {
        'proof/production two-phase': ['proof', 'preview'],
        'four-layer QA gate':       ['quality gate', 'four-layer'],
        'hash-bound approvals':     ['hash-bound', 'sha256'],
        'multi-episode V4':         ['season', 'series', 'multi-episode'],
        'WebSocket / live progress':['websocket', 'live progress'],
        'decoded visual QA':        ['visual qa', 'decoded', 'frame-level'],
        'overnight single-GPU':     ['overnight', 'scheduled'],
    }
    for capability, keywords in our_capabilities.items():
        if any(k in text for k in keywords):
            actions.append(f'potential capability overlap: {capability}')
    return actions


def render_findings(findings):
    today = datetime.now().strftime('%Y-%m-%d')
    if not findings:
        return f"## {today}\n\nNo new items across GitHub / Reddit / HN searches.\n\n"
    lines = [f"## {today}", '', f"Found **{len(findings)}** new item(s):", '']
    for f in findings:
        if f['type'] == 'github-repo':
            actions = compare_to_our_project(f)
            action_line = (' | 可能的差距: ' + '; '.join(actions)) if actions else ''
            lines.append(
                f"- **[{f['name']}]({f['url']})** — ⭐ {f['stars']} | {f.get('language') or 'n/a'}"
                f"\n  {f.get('description') or '(no description)'}{action_line}"
            )
        elif f['type'] == 'reddit-post':
            lines.append(
                f"- r/{f['subreddit']}: [{f['title']}]({f['url']})  \n  {f.get('summary') or ''}"
            )
        elif f['type'] == 'hn-post':
            lines.append(
                f"- HN ({f['query']}): [{f['title']}]({f['url']})  \n  {f.get('summary') or ''}"
            )
    return '\n'.join(lines) + '\n\n'


def append_runs(report):
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    header = ''
    if not RUNS.exists():
        header = '# Daily Research Runs\n\nDaily auto-scan results. 每 12 小时跑一次，09:00 与 21:00 (本地时间)。\n\n'
    with RUNS.open('a', encoding='utf-8') as f:
        if header:
            f.write(header)
        f.write(report)


def write_today(report):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    out = OUTPUTS / f'{today}-research.md'
    out.write_text(report, encoding='utf-8')


def push_to_repo(token, findings_count, dry_run):
    """Clone public repo, copy outputs, commit + push."""
    if dry_run:
        log(f'[dry-run] would push {findings_count} findings')
        return
    if REPO_DIR.exists():
        subprocess.run(['rmdir', '/S', '/Q', str(REPO_DIR)], shell=True, check=False)
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'clone', f'https://github.com/{REPO}.git', str(REPO_DIR)],
        capture_output=True, check=False
    )
    subprocess.run(['git', 'config', 'user.name', 'Dean Chen'], cwd=REPO_DIR, check=True)
    subprocess.run(['git', 'config', 'user.email', 'DeanChen85@users.noreply.github.com'],
                   cwd=REPO_DIR, check=True)

    # Copy research outputs to repo
    repo_research = REPO_DIR / 'docs' / 'research' / 'daily'
    repo_research.mkdir(parents=True, exist_ok=True)
    if OUTPUTS.exists():
        for f in OUTPUTS.glob('*.md'):
            (repo_research / f.name).write_bytes(f.read_bytes())
    # Copy RUNS.md
    repo_runs = REPO_DIR / 'docs' / 'research' / 'DAILY-RUNS.md'
    if RUNS.exists():
        repo_runs.write_bytes(RUNS.read_bytes())

    # Commit
    subprocess.run(['git', 'add', '-A'], cwd=REPO_DIR, check=False, capture_output=True)
    today = datetime.now().strftime('%Y-%m-%d')
    msg = f'research(daily): {today} — {findings_count} new items'
    subprocess.run(['git', 'commit', '-m', msg], cwd=REPO_DIR, check=False, capture_output=True)
    r = subprocess.run(['git', 'push', 'origin', 'main'], cwd=REPO_DIR,
                       capture_output=True, text=True)
    log(f'push: {r.stdout.strip()[:120] or r.stderr.strip()[:120]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Read-only: scan + report, do not write or push.')
    args = ap.parse_args()

    if not CONFIG.exists():
        log(f'config missing: {CONFIG}')
        sys.exit(1)

    config = load_config()
    state = load_state()
    findings = []

    # 1. GitHub
    token = get_github_token()
    log('scanning GitHub...')
    min_stars = config.get('minStarsForAlert', 0)
    for kw in config.get('githubKeywords', []):
        items = gh_search_repos(token, kw, config.get('githubDaysBack', 14))
        for r in items:
            if r.get('archived'):
                continue
            if (r.get('stargazers_count') or 0) < min_stars:
                continue
            key = r['full_name']
            if key in state['seenRepos']:
                continue
            findings.append({
                'type': 'github-repo',
                'name': key,
                'description': (r.get('description') or '')[:200],
                'stars': r.get('stargazers_count', 0),
                'url': r['html_url'],
                'language': r.get('language'),
                'created_at': r.get('created_at'),
            })
            state['seenRepos'][key] = r.get('created_at')

    # 2. Reddit
    log('scanning Reddit...')
    for sub in config.get('redditSubs', []):
        root = reddit_rss(sub)
        if root is None:
            continue
        for entry in getattr(root, 'entries', []):
            entry_id = entry.id
            if entry_id in state['seenPosts']:
                continue
            state['seenPosts'][entry_id] = datetime.now(timezone.utc).isoformat()
            findings.append({
                'type': 'reddit-post',
                'subreddit': sub,
                'title': entry.title,
                'url': entry.url,
                'summary': entry.summary,
            })

    # 2b. Hacker News (Algolia)
    log('scanning Hacker News...')
    for query in config.get('hnQueries', []):
        for entry in hn_search(query):
            entry_id = entry.id
            if entry_id in state['seenPosts']:
                continue
            state['seenPosts'][entry_id] = datetime.now(timezone.utc).isoformat()
            findings.append({
                'type': 'hn-post',
                'query': query,
                'title': entry.title,
                'url': entry.url,
                'summary': entry.summary,
            })

    log(f'{len(findings)} new findings')

    # 3. Write report
    report = render_findings(findings)
    if not args.dry_run:
        append_runs(report)
        write_today(report)
        save_state(state)
    push_to_repo(token, len(findings), args.dry_run)


if __name__ == '__main__':
    main()