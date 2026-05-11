"""
Scrape SHL Individual Test Solutions catalog → catalog.json
Uses Playwright — catalog is JS-rendered, Individual Test Solutions use tr[data-entity-id].
Run once: python scraper.py
"""
import json, time, concurrent.futures
from playwright.sync_api import sync_playwright

BASE    = "https://www.shl.com"
CATALOG = f"{BASE}/products/product-catalog/"
TYPE    = "1"   # 1 = Individual Test Solutions
UA      = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
WORKERS = 5     # parallel detail pages


def _extract_rows(page) -> list[dict]:
    return page.evaluate("""
        () => [...document.querySelectorAll('tr[data-entity-id]')].map(row => {
            const a    = row.querySelector('td.custom__table-heading__title a');
            if (!a) return null;
            const keys = [...row.querySelectorAll('span.product-catalogue__key')]
                            .map(s => s.innerText.trim()).filter(Boolean);
            const circles = [...row.querySelectorAll(
                'td.custom__table-heading__general span.catalogue__circle')];
            return {
                name:           a.innerText.trim(),
                href:           a.getAttribute('href'),
                test_types:     keys,
                remote_testing: circles[0]?.classList.contains('-yes') ?? false,
                adaptive:       circles[1]?.classList.contains('-yes') ?? false,
            };
        }).filter(Boolean)
    """)


def _next_start(page) -> int | None:
    return page.evaluate(f"""
        () => {{
            const next = [...document.querySelectorAll('a[href*="type={TYPE}"]')]
                            .find(a => a.innerText.trim() === 'Next');
            if (!next) return null;
            const m = next.href.match(/start=(\\d+)/);
            return m ? parseInt(m[1]) : null;
        }}
    """)


def _scrape_detail_worker(url: str) -> dict:
    """Each worker runs its own Playwright instance."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx     = browser.new_context(user_agent=UA)
        ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page    = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(300)
            result = page.evaluate("""
                () => {
                    const desc = document.querySelector(
                        '.product-hero__description, [class*="description"]');
                    const description = desc ? desc.innerText.trim().slice(0, 1200) : '';
                    let duration = '';
                    for (const el of document.querySelectorAll('*')) {
                        if (el.children.length === 0) {
                            const t = el.innerText || '';
                            if (/\\d+\\s*min/i.test(t) && t.length < 100) {
                                duration = t.trim(); break;
                            }
                        }
                    }
                    return {description, duration};
                }
            """)
        except Exception:
            result = {"description": "", "duration": ""}
        finally:
            browser.close()
    return result


def scrape_catalog() -> list[dict]:
    products = []
    seen     = set()

    # ── Phase 1: scrape all catalog pages (fast, sequential) ─────────────────
    with sync_playwright() as p:
        browser  = p.chromium.launch(headless=True)
        ctx      = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 900})
        ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        cat_page = ctx.new_page()

        print("Phase 1: scraping catalog pages…")
        start = 0
        while True:
            cat_page.goto(f"{CATALOG}?type={TYPE}&start={start}",
                          wait_until="networkidle", timeout=60000)
            cat_page.wait_for_timeout(800)

            rows = _extract_rows(cat_page)
            new  = [r for r in rows if r["name"] not in seen]
            for r in new:
                seen.add(r["name"])
                href = r.pop("href")
                r["url"]         = BASE + href if href.startswith("/") else href
                r["description"] = ""
                r["duration"]    = ""
            products.extend(new)
            print(f"  start={start:4d} → +{len(new):3d}  (total: {len(products)})")

            nxt = _next_start(cat_page)
            if nxt is None or nxt <= start:
                break
            start = nxt
            time.sleep(0.2)

        browser.close()

    # ── Phase 2: enrich detail pages in parallel ──────────────────────────────
    print(f"\nPhase 2: enriching {len(products)} products with {WORKERS} workers…")
    urls = [p["url"] for p in products]

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_scrape_detail_worker, url): i for i, url in enumerate(urls)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                detail = fut.result()
            except Exception:
                detail = {"description": "", "duration": ""}
            products[i]["description"] = detail["description"]
            products[i]["duration"]    = detail["duration"]
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(products)} enriched")

    return products


if __name__ == "__main__":
    data = scrape_catalog()
    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(data)} products → catalog.json")
