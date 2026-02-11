"""
NASCAR Odds Scraper - Action Network
=====================================
Scrapes NASCAR race odds from https://www.actionnetwork.com/nascar/odds
Stores odds for each driver across all available races in CSV format.

Requirements:
    pip install selenium webdriver-manager pandas beautifulsoup4

Usage:
    python nascar_odds_scraper.py

Output:
    - nascar_odds_output/nascar_all_odds_YYYYMMDD_HHMMSS.csv  (combined)
    - nascar_odds_output/<race-name>_odds_YYYYMMDD_HHMMSS.csv (per-race)
    - nascar_odds_output/debug_screenshot.png (for troubleshooting)
"""

import csv
import json
import os
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
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WDM = True
except ImportError:
    USE_WDM = False

# ── CONFIG ──────────────────────────────────────────────────────────────────
BASE_URL = "https://www.actionnetwork.com/nascar"
OUTPUT_DIR = Path("nascar_odds_output")
HEADLESS = True           # Runs silently in background for daily automation
WAIT_SECONDS = 6          # Wait for JS to render odds
SCROLL_PAUSE = 1.5        # Pause between scrolls to trigger lazy-load
# ────────────────────────────────────────────────────────────────────────────


def create_driver():
    """Create a Chrome WebDriver with anti-detection options."""
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    # Suppress automation flags
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    if USE_WDM:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)

    # Remove webdriver flag from navigator
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


def scroll_to_bottom(driver):
    """Scroll the page incrementally to trigger lazy-loaded content."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(15):
        driver.execute_script("window.scrollBy(0, 600);")
        time.sleep(SCROLL_PAUSE)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)


def dismiss_popups(driver):
    """Close cookie banners, modals, or ad overlays."""
    popup_selectors = [
        "button[aria-label='Close']",
        "button[aria-label='close']",
        "[class*='close-button']",
        "[class*='modal'] button",
        "#onetrust-accept-btn-handler",
    ]
    for sel in popup_selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            if btn.is_displayed():
                btn.click()
                time.sleep(0.5)
        except Exception:
            pass


def get_race_options(driver):
    """
    Detect available races from the page's dropdown/select element.
    Returns list of dicts: [{"name": "DAYTONA 500", "index": 0}, ...]
    """
    races = []

    # Method 1: <select> elements
    try:
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            options = sel.find_elements(By.TAG_NAME, "option")
            for i, opt in enumerate(options):
                text = opt.text.strip()
                value = opt.get_attribute("value") or ""
                if text and text.lower() not in ("select", "choose", ""):
                    races.append({
                        "name": text, "value": value,
                        "index": i, "type": "select", "select_id": sel.id
                    })
            if races:
                print(f"  Found {len(races)} races via <select> dropdown")
                return races
    except Exception:
        pass

    # Method 2: Custom React dropdown
    try:
        dropdown_candidates = driver.find_elements(
            By.CSS_SELECTOR,
            "[class*='dropdown'], [class*='select'], [role='combobox'], "
            "[class*='picker'], [data-testid*='select'], [data-testid*='dropdown']"
        )
        for dd in dropdown_candidates:
            text = dd.text.strip()
            if any(kw in text.upper() for kw in ["500", "RACE", "CUP", "SPEEDWAY", "MOTOR", "DAYTONA"]):
                dd.click()
                time.sleep(1.5)

                for opt_sel in ["[role='option']", "[class*='option']", "[class*='menu-item']", "li"]:
                    items = driver.find_elements(By.CSS_SELECTOR, opt_sel)
                    visible = [it for it in items if it.is_displayed() and it.text.strip()]
                    if len(visible) > 1:
                        for idx, item in enumerate(visible):
                            races.append({"name": item.text.strip(), "index": idx, "type": "custom"})
                        break

                try:
                    webdriver.ActionChains(driver).send_keys("\ue00c").perform()
                except Exception:
                    pass
                time.sleep(0.5)

                if races:
                    print(f"  Found {len(races)} races via custom dropdown")
                    return races
    except Exception:
        pass

    # Fallback
    print("  Could not detect race dropdown - scraping current page")
    try:
        heading = driver.find_element(By.CSS_SELECTOR, "h1, h2, [class*='title']")
        race_name = heading.text.strip()
    except Exception:
        race_name = "Current Race"
    return [{"name": race_name, "index": 0, "type": "none"}]


def select_race(driver, race):
    """Select a race from the dropdown."""
    try:
        if race.get("type") == "select":
            from selenium.webdriver.support.ui import Select
            selects = driver.find_elements(By.TAG_NAME, "select")
            for sel in selects:
                options = sel.find_elements(By.TAG_NAME, "option")
                for opt in options:
                    if opt.text.strip() == race["name"]:
                        select_obj = Select(sel)
                        select_obj.select_by_visible_text(race["name"])
                        time.sleep(WAIT_SECONDS)
                        return True

        elif race.get("type") == "custom":
            dropdown_candidates = driver.find_elements(
                By.CSS_SELECTOR,
                "[class*='dropdown'], [class*='select'], [role='combobox'], [class*='picker']"
            )
            for dd in dropdown_candidates:
                try:
                    if dd.is_displayed():
                        dd.click()
                        time.sleep(1)
                        break
                except Exception:
                    continue

            options = driver.find_elements(By.CSS_SELECTOR, "[role='option'], [class*='option'], li")
            for opt in options:
                if opt.text.strip() == race["name"] and opt.is_displayed():
                    opt.click()
                    time.sleep(WAIT_SECONDS)
                    return True

        return False
    except Exception as e:
        print(f"  Warning selecting race: {e}")
        return False


def detect_sportsbooks(soup):
    """Detect sportsbook names from the header area."""
    KNOWN_BOOKS = [
        "BetMGM", "FanDuel", "DraftKings", "Caesars",
        "BetRivers", "ESPN BET", "Fanatics", "Hard Rock",
        "PointsBet", "BetUS", "Bovada"
    ]
    found = []
    for img in soup.find_all("img"):
        alt = (img.get("alt") or img.get("title") or "").lower()
        for book in KNOWN_BOOKS:
            if book.lower().replace(" ", "") in alt.lower().replace(" ", ""):
                if book not in found:
                    found.append(book)
    if not found:
        page_text = soup.get_text().upper()
        for book in KNOWN_BOOKS:
            if book.upper() in page_text and book not in found:
                found.append(book)
    return found if found else KNOWN_BOOKS[:6]


def extract_driver_from_element(element, sportsbooks):
    """Extract driver name and odds from a BeautifulSoup element."""
    text = element.get_text(separator="|", strip=True)

    name = ""
    for a_tag in element.find_all("a"):
        a_text = a_tag.get_text(strip=True)
        if a_text and not re.match(r"^[+-]?\d+$", a_text) and len(a_text) > 2:
            name = a_text
            break
    if not name:
        for child in element.find_all(["span", "div", "p", "strong", "h3", "h4"]):
            child_text = child.get_text(strip=True)
            if (child_text and not re.match(r"^[+-]?\d+$", child_text)
                    and len(child_text) > 2
                    and not re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s", child_text)):
                name = child_text
                break
    if not name:
        return None

    odds_values = re.findall(r"[+-]\d{3,}", text)
    if len(odds_values) < 2:
        return None

    driver = {"driver_name": name}
    driver["open_odds"] = odds_values[0]
    driver["best_odds"] = odds_values[1]
    book_odds = odds_values[2:]
    for i, book in enumerate(sportsbooks):
        driver[book] = book_odds[i] if i < len(book_odds) else ""
    return driver


def selenium_text_extraction(driver, sportsbooks):
    """
    Fallback: parse the visible page text line by line.
    Works when React renders content not in traditional HTML structure.
    """
    drivers = []
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        full_text = body.text
        lines = full_text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Check if line looks like a driver name
            skip_words = [
                "SCHEDULED", "OPEN", "BEST", "ODDS", "NASCAR", "OUTRIGHT",
                "WINNER", "SPORTSBOOK", "BETMGM", "FANDUEL", "DRAFTKINGS",
                "CAESARS", "BETRIVERS", "ESPN", "FANATICS"
            ]
            is_date = bool(re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s", line))
            is_odds = bool(re.match(r"^[+-]\d{3,}$", line))
            is_skip = any(sw in line.upper() for sw in skip_words)

            if (line and len(line) > 3 and len(line) < 50
                    and not is_date and not is_odds and not is_skip
                    and not re.match(r"^\d+$", line)):

                # Collect odds from subsequent lines
                odds = []
                j = i + 1
                while j < len(lines) and j < i + 30:
                    next_line = lines[j].strip()
                    odds_match = re.match(r"^([+-]\d{3,})$", next_line)
                    if odds_match:
                        odds.append(odds_match.group(1))
                    elif re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s", next_line):
                        pass  # Skip date lines
                    elif next_line and len(odds) >= 3:
                        # Probably hit the next driver name
                        break
                    j += 1

                if len(odds) >= 3:
                    d = {"driver_name": line}
                    d["open_odds"] = odds[0]
                    d["best_odds"] = odds[1]
                    for k, book in enumerate(sportsbooks):
                        d[book] = odds[k + 2] if k + 2 < len(odds) else ""
                    drivers.append(d)
                    i = j
                    continue
            i += 1
    except Exception as e:
        print(f"  Text extraction error: {e}")
    return drivers


def extract_from_next_data(data):
    """Recursively search __NEXT_DATA__ for odds arrays."""
    results = []

    def search(obj, depth=0):
        if depth > 15:
            return
        if isinstance(obj, dict):
            name = obj.get("full_name") or obj.get("name") or obj.get("display_name") or ""
            if name and ("odds" in obj or "books" in obj or "lines" in obj):
                driver = {"driver_name": name}
                odds_data = obj.get("odds") or obj.get("books") or obj.get("lines") or {}
                if isinstance(odds_data, dict):
                    for book, val in odds_data.items():
                        if isinstance(val, dict):
                            driver[book] = str(val.get("odds", val.get("line", "")))
                        else:
                            driver[book] = str(val)
                results.append(driver)
            else:
                for v in obj.values():
                    search(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                search(item, depth + 1)

    search(data)
    return results


def parse_odds_table(driver):
    """
    Parse the odds table from the current page.
    Returns (sportsbook_headers, list_of_driver_dicts).
    """
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # ── Try __NEXT_DATA__ first (fastest & most reliable) ──
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data:
        try:
            data = json.loads(next_data.string)
            drivers = extract_from_next_data(data)
            if drivers:
                print(f"  Extracted {len(drivers)} drivers from __NEXT_DATA__")
                books = list(set(
                    k for d in drivers for k in d.keys()
                ) - {"driver_name", "scheduled", "open_odds", "best_odds"})
                return sorted(books), drivers
        except Exception:
            pass

    # ── DOM-based parsing ──
    sportsbooks = detect_sportsbooks(soup)
    drivers = []

    # Find elements containing odds patterns
    all_odds_strings = soup.find_all(string=re.compile(r"^[+-]\d{3,}$"))
    if all_odds_strings:
        seen_parents = set()
        for el in all_odds_strings:
            parent = el.parent
            for _ in range(10):
                if parent is None:
                    break
                pid = id(parent)
                if pid in seen_parents:
                    break
                text = parent.get_text(separator="|")
                odds_count = len(re.findall(r"[+-]\d{3,}", text))
                if odds_count >= 3:
                    seen_parents.add(pid)
                    d = extract_driver_from_element(parent, sportsbooks)
                    if d:
                        drivers.append(d)
                    break
                parent = parent.parent

    # ── Fallback: text-based extraction ──
    if not drivers:
        print("  DOM parsing found nothing, trying text-based extraction...")
        drivers = selenium_text_extraction(driver, sportsbooks)

    # Deduplicate
    seen = set()
    unique = []
    for d in drivers:
        name = d.get("driver_name", "")
        if name and name not in seen:
            seen.add(name)
            unique.append(d)

    print(f"  Parsed {len(unique)} drivers")
    return sportsbooks, unique


def scrape_all_races():
    """Main entry point."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("NASCAR Odds Scraper - Action Network")
    print("=" * 60)

    driver = create_driver()

    try:
        print(f"\nNavigating to {BASE_URL} ...")
        driver.get(BASE_URL)
        time.sleep(WAIT_SECONDS)

        dismiss_popups(driver)

        # Debug screenshot
        ss_path = OUTPUT_DIR / "debug_screenshot.png"
        driver.save_screenshot(str(ss_path))
        print(f"  Debug screenshot: {ss_path}")

        # Scroll to load all drivers
        print("  Scrolling to load all drivers...")
        scroll_to_bottom(driver)

        # Detect races
        print("\nDetecting available races...")
        races = get_race_options(driver)
        print(f"  Races: {[r['name'] for r in races]}")

        for i, race in enumerate(races):
            race_name = race["name"]
            print(f"\n--- [{i + 1}/{len(races)}] {race_name} ---")

            if i > 0:
                if not select_race(driver, race):
                    print("  Could not select race, skipping")
                    continue
                time.sleep(WAIT_SECONDS)
                scroll_to_bottom(driver)

            sportsbooks, drivers = parse_odds_table(driver)

            if not drivers:
                print("  No drivers found! Check debug screenshot.")
                driver.save_screenshot(
                    str(OUTPUT_DIR / f"debug_{re.sub(r'[^a-z0-9]', '_', race_name.lower())}.png")
                )
                continue

            print(f"  {len(drivers)} drivers | Books: {sportsbooks}")

            for d in drivers:
                d["race"] = race_name
                d["scraped_at"] = datetime.now().isoformat()
                all_data.append(d)

            # Per-race CSV
            slug = re.sub(r"[^a-z0-9]+", "-", race_name.lower()).strip("-")
            race_csv = OUTPUT_DIR / f"{slug}_odds_{timestamp}.csv"
            pd.DataFrame(drivers).to_csv(race_csv, index=False)
            print(f"  Saved: {race_csv}")

    finally:
        driver.quit()

    # Combined CSV
    if all_data:
        combined_csv = OUTPUT_DIR / f"nascar_all_odds_{timestamp}.csv"
        df = pd.DataFrame(all_data)
        priority = ["race", "driver_name", "open_odds", "best_odds", "scraped_at"]
        cols = [c for c in priority if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]
        df.to_csv(combined_csv, index=False)

        print(f"\n{'=' * 60}")
        print(f"DONE! Combined CSV: {combined_csv}")
        race_count = len(set(d["race"] for d in all_data))
        print(f"Total: {len(all_data)} drivers across {race_count} race(s)")
        print(f"{'=' * 60}")
        print(f"\nPreview:")
        print(df.head(10).to_string(index=False))
    else:
        print("\nNo data scraped!")
        print("Troubleshooting:")
        print("  1. Set HEADLESS = False to watch the browser")
        print("  2. Check debug_screenshot.png in nascar_odds_output/")
        print("  3. The site may need a US IP address")
        print("  4. Try increasing WAIT_SECONDS to 10")

    return all_data


if __name__ == "__main__":
    scrape_all_races() 
