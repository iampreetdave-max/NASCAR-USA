"""
NASCAR Odds Scraper - Action Network (All Races × All Markets)
===============================================================
URL: https://www.actionnetwork.com/nascar

Strategy: One full pass through ALL races per market type.
  Pass 1: Outright Winner → all 42 races
  Pass 2: Top 3 → all 42 races  
  Pass 3: Top 5 → all 42 races
  Pass 4: Top 10 → all 42 races
  Pass 5: Top 20 → all 42 races
Fresh browser for each pass to avoid Chrome memory crash.

Requirements:  pip install selenium webdriver-manager pandas beautifulsoup4
Usage:         python nascar_odds_scraper.py
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WDM = True
except ImportError:
    USE_WDM = False

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_URL = "https://www.actionnetwork.com/nascar"
OUTPUT_DIR = Path("nascar_odds_output")
HEADLESS = True
WAIT_SECONDS = 6
SCROLL_PAUSE = 1.5

MARKETS = [
    {"label": "winner", "text": "Outright Winner"},
    {"label": "top3",   "text": "Top 3"},
    {"label": "top5",   "text": "Top 5"},
    {"label": "top10",  "text": "Top 10"},
    {"label": "top20",  "text": "Top 20"},
]
# ────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("nascar")


# ═══════════════════════════════════════════════════════════════════════════
# BROWSER
# ═══════════════════════════════════════════════════════════════════════════

def create_driver():
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    if USE_WDM:
        service = Service(ChromeDriverManager().install())
        drv = webdriver.Chrome(service=service, options=options)
    else:
        drv = webdriver.Chrome(options=options)

    drv.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return drv


def scroll_to_bottom(drv):
    last_h = drv.execute_script("return document.body.scrollHeight")
    for _ in range(15):
        drv.execute_script("window.scrollBy(0, 600);")
        time.sleep(SCROLL_PAUSE)
        new_h = drv.execute_script("return document.body.scrollHeight")
        if new_h == last_h:
            break
        last_h = new_h
    drv.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)


def dismiss_popups(drv):
    for sel in ["#onetrust-accept-btn-handler", "button[aria-label='Close']",
                "button[aria-label='close']", "[class*='close-button']"]:
        try:
            btn = drv.find_element(By.CSS_SELECTOR, sel)
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# DROPDOWN DETECTION
# ═══════════════════════════════════════════════════════════════════════════

RACE_KW = ["500", "SPEEDWAY", "RACEWAY", "DAYTONA", "ATLANTA", "BRISTOL",
           "CHARLOTTE", "DOVER", "TALLADEGA", "MARTINSVILLE", "PHOENIX",
           "LAS VEGAS", "MOTOR", "INTERNATIONAL", "DUEL", "CLASH",
           "CHAMPIONSHIP", "ALL-STAR", "GRAND PRIX", "ROVAL"]
MARKET_KW = ["OUTRIGHT WINNER", "TOP 3", "TOP 5", "TOP 10", "TOP 20", "MATCHUPS"]


def find_dropdowns(drv):
    """Find race and market <select> elements by option content."""
    race_sel = None
    market_sel = None
    for sel in drv.find_elements(By.TAG_NAME, "select"):
        try:
            opts = " ".join(o.text.strip().upper() for o in sel.find_elements(By.TAG_NAME, "option"))
            if any(kw in opts for kw in MARKET_KW) and market_sel is None:
                market_sel = sel
            elif any(kw in opts for kw in RACE_KW) and race_sel is None:
                race_sel = sel
        except Exception:
            pass
    return race_sel, market_sel


def get_options(sel_element):
    return [o.text.strip() for o in sel_element.find_elements(By.TAG_NAME, "option") if o.text.strip()]


def safe_select(drv, sel_element, text):
    """Select option, handle stale elements by re-finding."""
    try:
        Select(sel_element).select_by_visible_text(text)
        time.sleep(WAIT_SECONDS)
        return True
    except Exception:
        pass
    # Re-find and retry
    try:
        race_sel, market_sel = find_dropdowns(drv)
        for candidate in [race_sel, market_sel]:
            if candidate is None:
                continue
            if text in get_options(candidate):
                Select(candidate).select_by_visible_text(text)
                time.sleep(WAIT_SECONDS)
                return True
    except Exception as e:
        log.warning(f"    Failed to select '{text}': {e}")
    return False


# ═══════════════════════════════════════════════════════════════════════════
# ODDS PARSING
# ═══════════════════════════════════════════════════════════════════════════

def detect_sportsbooks(soup):
    KNOWN = ["BetMGM", "FanDuel", "DraftKings", "Caesars", "BetRivers",
             "ESPN BET", "Fanatics", "Hard Rock", "PointsBet"]
    found = []
    for img in soup.find_all("img"):
        alt = (img.get("alt") or img.get("title") or "").lower()
        for book in KNOWN:
            if book.lower().replace(" ", "") in alt.replace(" ", "") and book not in found:
                found.append(book)
    return found if found else KNOWN[:6]


def scrape_table(drv):
    """Parse visible odds table. Returns (drivers_list, sportsbooks)."""
    soup = BeautifulSoup(drv.page_source, "html.parser")
    sportsbooks = detect_sportsbooks(soup)

    nd = soup.find("script", id="__NEXT_DATA__")
    if nd:
        try:
            data = json.loads(nd.string)
            drivers = _json_extract(data, sportsbooks)
            if drivers:
                return drivers, sportsbooks
        except Exception:
            pass

    return _text_extract(drv, sportsbooks), sportsbooks


def _text_extract(drv, sportsbooks):
    drivers = []
    try:
        lines = drv.find_element(By.TAG_NAME, "body").text.split("\n")
    except Exception:
        return drivers

    SKIP = {
        "SCHEDULED", "OPEN", "BEST", "ODDS", "NASCAR", "OUTRIGHT",
        "WINNER", "SPORTSBOOK", "BETMGM", "FANDUEL", "DRAFTKINGS",
        "CAESARS", "BETRIVERS", "ESPN", "FANATICS", "MATCHUPS",
        "TOP 3", "TOP 5", "TOP 10", "TOP 20", "HARD ROCK",
        "NASCAR ODDS", "NASCAR ODDS & BETTING LINES",
        "ODDS SETTINGS", "SELECT A LOCATION", "OFFERS",
        "GET APP", "TRY PRO NOW", "SIGN IN", "SUBSCRIBE",
        "NO CODE NEEDED", "PROMO CODE", "CLAIM",
        "ALL MARKETS", "SPREAD", "TOTAL", "MONEYLINE",
        "NFL", "NBA", "NCAAB", "NCAAF", "NHL", "MLB", "SOCCER",
        "WNBA", "UFC", "ATP", "WTA", "NCAAW",
    }

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or len(line) <= 2 or len(line) >= 50:
            i += 1
            continue

        is_odds = bool(re.match(r"^[+-]?\d{2,}$", line))
        is_date = bool(re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s", line))
        is_skip = line.upper() in SKIP
        is_num = bool(re.match(r"^\d+$", line))
        is_na = line.upper() == "N/A"

        if not is_odds and not is_date and not is_skip and not is_num and not is_na:
            scheduled = ""
            odds = []
            j = i + 1
            while j < len(lines) and j < i + 40:
                nxt = lines[j].strip()
                if re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s", nxt):
                    scheduled = nxt
                elif re.match(r"^[+-]?\d{2,}$", nxt):
                    odds.append(nxt)
                elif nxt.upper() == "N/A":
                    odds.append("")
                elif nxt == "":
                    pass
                else:
                    if len(odds) >= 2:
                        break
                j += 1

            if len(odds) >= 2:
                d = {
                    "driver_name": line,
                    "scheduled": scheduled,
                    "open": odds[0],
                    "best": odds[1],
                    "books": {},
                }
                for k, book in enumerate(sportsbooks):
                    d["books"][book] = odds[k + 2] if k + 2 < len(odds) else ""
                drivers.append(d)
                i = j
                continue
        i += 1
    return drivers


def _json_extract(data, sportsbooks):
    results = []
    def search(obj, depth=0):
        if depth > 15:
            return
        if isinstance(obj, dict):
            name = obj.get("full_name") or obj.get("name") or obj.get("display_name") or ""
            if name and ("odds" in obj or "books" in obj or "lines" in obj):
                d = {"driver_name": name, "scheduled": "", "open": "", "best": "", "books": {}}
                od = obj.get("odds") or obj.get("books") or obj.get("lines") or {}
                if isinstance(od, dict):
                    for bk, val in od.items():
                        v = val.get("odds", val.get("line", "")) if isinstance(val, dict) else val
                        v = str(v) if v and str(v).upper() != "N/A" else ""
                        d["books"][bk] = v
                results.append(d)
            else:
                for v in obj.values():
                    search(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                search(item, depth + 1)
    search(data)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE PASS: one market, all races
# ═══════════════════════════════════════════════════════════════════════════

def scrape_one_market(market_label, market_text, combined, all_book_names):
    """
    Open a fresh browser, select the market, loop through all races.
    Merges results into the shared 'combined' dict.
    """
    log.info(f"\n{'='*60}")
    log.info(f"PASS: {market_text} ({market_label})")
    log.info(f"{'='*60}")

    drv = create_driver()
    try:
        drv.get(BASE_URL)
        time.sleep(WAIT_SECONDS + 2)
        dismiss_popups(drv)
        scroll_to_bottom(drv)

        race_sel, market_sel = find_dropdowns(drv)
        if not race_sel:
            log.error("  No race dropdown found!")
            return
        if not market_sel:
            log.error("  No market dropdown found!")
            return

        race_options = get_options(race_sel)
        log.info(f"  {len(race_options)} races found")

        # Select the market first
        log.info(f"  Selecting market: {market_text}")
        if not safe_select(drv, market_sel, market_text):
            log.error(f"  Could not select market '{market_text}', skipping entire pass")
            return

        # Now loop through all races
        for ri, race_name in enumerate(race_options):
            log.info(f"  [{ri+1}/{len(race_options)}] {race_name}")

            # Select race
            race_sel_fresh, _ = find_dropdowns(drv)
            if race_sel_fresh is None:
                log.warning(f"    Lost race dropdown, skipping")
                continue

            if not safe_select(drv, race_sel_fresh, race_name):
                log.warning(f"    Could not select race, skipping")
                continue

            # Re-select market (it may reset after race change)
            _, market_sel_fresh = find_dropdowns(drv)
            if market_sel_fresh:
                try:
                    current_market = Select(market_sel_fresh).first_selected_option.text.strip()
                    if current_market != market_text:
                        log.info(f"    Market reset to '{current_market}', re-selecting...")
                        safe_select(drv, market_sel_fresh, market_text)
                except Exception:
                    safe_select(drv, market_sel_fresh, market_text)

            scroll_to_bottom(drv)
            drivers, sportsbooks = scrape_table(drv)
            all_book_names.update(sportsbooks)

            if not drivers:
                log.warning(f"    No drivers")
                continue

            log.info(f"    {len(drivers)} drivers")

            # Merge into combined
            for d in drivers:
                key = (race_name, d["driver_name"])
                if key not in combined:
                    combined[key] = {
                        "race": race_name,
                        "driver_name": d["driver_name"],
                        "scheduled": d.get("scheduled", ""),
                    }

                row = combined[key]
                if d.get("scheduled"):
                    row["scheduled"] = d["scheduled"]

                o = d.get("open", "")
                b = d.get("best", "")
                row[f"{market_label}_open"] = o if str(o).upper() != "N/A" else ""
                row[f"{market_label}_best"] = b if str(b).upper() != "N/A" else ""

                for book, val in d.get("books", {}).items():
                    row[f"{market_label}_{book}"] = val if val and str(val).upper() != "N/A" else ""

    except Exception as e:
        log.error(f"  Pass failed: {e}")
    finally:
        try:
            drv.quit()
        except Exception:
            pass

    log.info(f"  Pass complete. Total rows so far: {len(combined)}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def scrape_all():
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log.info("=" * 60)
    log.info("NASCAR Odds Scraper — All Races x All Markets")
    log.info(f"URL: {BASE_URL}")
    log.info("One fresh browser per market pass (avoids Chrome crash)")
    log.info("=" * 60)

    combined = {}
    all_book_names = set()

    for market in MARKETS:
        scrape_one_market(market["label"], market["text"], combined, all_book_names)

    # ── Build CSV ──
    if not combined:
        log.error("No data scraped!")
        return

    rows = list(combined.values())
    df = pd.DataFrame(rows)

    sorted_books = sorted(all_book_names)
    ordered = ["race", "driver_name", "scheduled"]
    for m in MARKETS:
        lbl = m["label"]
        ordered.append(f"{lbl}_open")
        ordered.append(f"{lbl}_best")
        for bk in sorted_books:
            ordered.append(f"{lbl}_{bk}")

    final = [c for c in ordered if c in df.columns]
    extra = [c for c in df.columns if c not in final]
    df = df[final + extra]
    df["scraped_at"] = datetime.now().isoformat()

    csv_path = OUTPUT_DIR / f"nascar_all_odds_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"DONE!")
    log.info(f"  File: {csv_path}")
    log.info(f"  Rows: {len(df)} | Races: {df['race'].nunique()} | Cols: {len(df.columns)}")
    log.info(f"{'='*60}")
    log.info(f"\nPreview:\n{df.head(3).to_string(index=False)}")


if __name__ == "__main__":
    scrape_all()


    
