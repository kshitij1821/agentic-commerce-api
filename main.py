import os
import uuid
import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import GoogleSearch
from fastapi.middleware.cors import CORSMiddleware

# Safe imports for optional dependencies
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

try:
    import pgeocode
except ImportError:
    pgeocode = None

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

try:
    from forex_python.converter import CurrencyRates
except ImportError:
    CurrencyRates = None

# Load environment variables
load_dotenv()

# Initialize API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY not found in .env file.")
if not SERPAPI_KEY:
    print("⚠️ WARNING: SERPAPI_KEY not found in .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

# CORS: allow_credentials must be False when allow_origins is "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DATA MODELS
# ==========================================

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    json_output: dict | None = None

# ==========================================
# SESSION MANAGEMENT
# ==========================================

sessions: Dict[str, Dict[str, Any]] = {}

# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are an AI shopping assistant.

Your ONLY goal is to help users plan purchases.

Guardrails:
- Refuse unrelated requests politely.
- Keep responses concise (1–2 sentences).
- Ask only one question at a time.
- Guide users toward purchase planning.
- Do not engage in chit-chat.

Conversation Flow:
1. Understand items user wants.
2. Collect budget constraints and ask for preferences.
3. Ask for delivery place and deadline.
4. Summarize order.
5. Ask for confirmation.
6. Only after confirmation output FINAL_JSON.

Defaults:
- If delivery deadline is not specified even after being asked, assume 14 days from today.
- If item preferences are not specified even after being asked, "No specific preferences".

If the user asks for general categories, you MUST break it down into specific searchable categories.
Example - "electronics" -> "laptop", "mouse", "keyboard"

When user confirms, respond:

FINAL_JSON:
<json>

JSON format:
{
  "delivery_pincode": "...",
  "delivery_deadline_date": "...",
  "total_budget" : number,
  "items": [
    {
      "query": "...",
      "preferences": "..."
    }
  ]
}

Rules:
- NEVER output FINAL_JSON before confirmation.
- Keep answers short.
- Stay within shopping assistance.
- NEVER leave fields null; apply defaults if needed.
"""

# ==========================================
# HELPER FUNCTIONS (Location, Currency, Date)
# ==========================================

def _get_nomi(country):
    if pgeocode:
        try:
            return pgeocode.Nominatim(country)
        except Exception:
            pass
    return None

def _simplify_location(place: str, state: str, country_code: str) -> str:
    """Reduce to city + state + country for SerpApi."""
    city_match = re.search(r"\(([^)]+)\)", place) if place else None
    if city_match:
        city = city_match.group(1).strip()
    else:
        city = place.split(",")[0].strip() if place else ""

    if not city:
        city = state or ("India" if country_code == "in" else "United States")

    state = (state or "").strip()
    if country_code == "in":
        return f"{city}, {state}, India".strip(", ")
    return f"{city}, {state}, United States".strip(", ")

def get_location_from_pincode(pincode: str) -> tuple:
    """Maps pincode -> (Location String, Currency, Country Code)."""
    pincode = str(pincode).strip()
    country = "in" if len(pincode) == 6 and pincode.isdigit() else "us"
    nomi = _get_nomi(country)

    if nomi:
        try:
            result = nomi.query_postal_code(pincode)
            if result is not None:
                place = getattr(result, "place_name", None)
                state = getattr(result, "state_name", None)
                if hasattr(result, 'values'):
                    place = str(place) if place and str(place) != 'nan' else ""
                    state = str(state) if state and str(state) != 'nan' else ""
                if place or state:
                    loc = _simplify_location(place, state, country)
                    currency = "INR" if country == "in" else "USD"
                    return (loc, currency, country)
        except Exception as e:
            print(f"⚠️ Location lookup warning: {e}")

    return ("United States", "USD", "us") if country == "us" else ("India", "INR", "in")

def convert_budget_to_local(amount: float, from_curr: str, to_curr: str) -> float:
    if from_curr == to_curr:
        return amount
    if CurrencyRates:
        try:
            c = CurrencyRates()
            return c.convert(from_curr, to_curr, amount)
        except Exception:
            pass
    rates = {"USD": 1.0, "INR": 84.0, "EUR": 0.92, "GBP": 0.79}
    try:
        usd_base = amount / rates.get(from_curr, 1.0)
        return round(usd_base * rates.get(to_curr, 1.0), 2)
    except Exception:
        return amount

def parse_delivery_date(delivery_str: str, base_date: datetime):
    if not delivery_str or "unknown" in str(delivery_str).lower():
        return None
    s = str(delivery_str).lower()
    if "tomorrow" in s:
        return (base_date + timedelta(days=1)).strftime("%Y-%m-%d")
    if "today" in s:
        return base_date.strftime("%Y-%m-%d")
    try:
        match = re.search(r"([a-z]{3})\s(\d{1,2})", s)
        if match:
            dt = date_parser.parse(f"{match.group(1)} {match.group(2)} {base_date.year}")
            if dt < base_date - timedelta(days=30):
                dt = dt.replace(year=base_date.year + 1)
            return dt.strftime("%Y-%m-%d")
        if date_parser:
            dt = date_parser.parse(s, fuzzy=True, default=base_date)
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None

# ==========================================
# SMART SHOPPING AGENT
# ==========================================

class SmartShoppingAgent:
    def __init__(self):
        self.serp_api_key = SERPAPI_KEY
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def search_google_shopping(self, query, pincode):
        location, currency, gl = get_location_from_pincode(pincode)
        params = {
            "api_key": self.serp_api_key,
            "engine": "google_shopping",
            "q": query,
            "location": location,
            "google_domain": "google.com",
            "gl": gl,
            "hl": "en",
            "num": 10
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            items = results.get("shopping_results", [])
            clean_items = []
            for item in items:
                price_str = item.get("price", "0")
                price_val = 0.0
                if price_str:
                    clean_p = re.sub(r'[^\d.]', '', str(price_str).replace(',', ''))
                    try:
                        price_val = float(clean_p)
                    except ValueError:
                        pass
                clean_items.append({
                    "title": item.get("title"),
                    "price_str": price_str,
                    "price_val": price_val,
                    "currency": currency,
                    "source": item.get("source", "Unknown"),
                    "link": item.get("link"),
                    "delivery": item.get("delivery", "Unknown"),
                    "thumbnail": item.get("thumbnail"),
                    "image_link": item.get("thumbnail") or item.get("link"),
                    "rating": item.get("rating", "N/A"),
                    "reviews": item.get("reviews", 0)
                })
            return clean_items
        except Exception as e:
            print(f"❌ Search Error: {e}")
            return []

    def enrich_product_link(self, product):
        """Replace Google redirect / broken links with direct retailer URLs via site: search."""
        original_link = product.get("link") or ""
        source = (product.get("source") or "").lower()
        title = (product.get("title") or "").strip()
        is_bad_link = (
            not original_link
            or "google.com" in original_link
            or "/aclk" in original_link
            or "google.co.in" in original_link
        )
        if not is_bad_link:
            return product
        if not title or not self.serp_api_key:
            return product
        # Build site-specific query to get real product URL
        site_map = {
            "decathlon": "decathlon.in",
            "amazon": "amazon.in",
            "flipkart": "flipkart.com",
            "myntra": "myntra.com",
        }
        site = None
        for k, v in site_map.items():
            if k in source:
                site = v
                break
        if not site:
            site = source.replace(" ", "").lower() + ".com" if source else "amazon.in"
        search_query = f"{title} site:{site}"
        params = {
            "api_key": self.serp_api_key,
            "engine": "google",
            "q": search_query,
            "num": 3,
        }
        try:
            search = GoogleSearch(params)
            res = search.get_dict()
            organic = res.get("organic_results", [])
            for hit in organic:
                link = hit.get("link")
                if link and "google.com" not in link and "/aclk" not in link:
                    product["link"] = link
                    if "delivery" not in str(product.get("delivery", "")).lower():
                        snippet = hit.get("snippet", "")
                        product["delivery"] = f"Check site — {snippet[:50]}..." if snippet else "Check retailer"
                    return product
        except Exception as e:
            print(f"⚠️ Enrich link warning: {e}")
        return product

    def rank_and_decide(self, query, products, constraints):
        max_price = constraints.get("max_price", float('inf'))
        currency = constraints.get("currency", "USD")
        valid_products = [p for p in products if p.get("price_val", 0) <= max_price]
        if not valid_products:
            valid_products = sorted(products, key=lambda x: x.get("price_val", 0))[:5]
        retailers = list(set([p.get("source", "Unknown") for p in valid_products]))
        retailers_str = ", ".join(retailers[:10])
        # Ensure each product has image_link for LLM output (frontend uses image_link or thumbnail)
        for p in valid_products:
            if "image_link" not in p or not p.get("image_link"):
                p["image_link"] = p.get("thumbnail") or ""
        prompt = f"""
You are a shopping assistant with STRICT budget enforcement.
GOAL: Select the best 3 options for "{query}" from DIFFERENT RETAILERS within budget.
Use COMPARATIVE reasoning (e.g. "Best Pick: X because... Runner-up: Y because... Budget Pick: Z because...").

CRITICAL: Use ONLY the "link" and "image_link" values from the PRODUCTS list below. Do not invent or change URLs.
BUDGET: {max_price} {currency} per item.
RETAILER DIVERSITY: All 3 picks MUST be from DIFFERENT retailers. Available: {retailers_str}

CONTEXT:
- Location: {constraints.get('location')}
- Budget: {max_price} {currency}
- Deadline: {constraints.get('deadline')}
- Preferences: {constraints.get('preferences')}

PRODUCTS (use their "link" and "image_link" exactly):
{json.dumps(valid_products, indent=2)}

OUTPUT valid JSON only:
{{
  "best_pick": {{ "title": "...", "price": "...", "source": "...", "link": "<use exact link from products>", "image_link": "<use exact image_link from products>", "delivery": "...", "reasoning": "..." }},
  "runner_up": {{ "title": "...", "price": "...", "source": "...", "link": "<use exact link from products>", "image_link": "<use exact image_link from products>", "delivery": "...", "reasoning": "..." }},
  "budget_pick": {{ "title": "...", "price": "...", "source": "...", "link": "<use exact link from products>", "image_link": "<use exact image_link from products>", "delivery": "...", "reasoning": "..." }}
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Return valid JSON only. Use only link and image_link values from the provided PRODUCTS."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            out = json.loads(response.choices[0].message.content)
            # Ensure we keep real links from our enriched products (LLM sometimes drops or wrong copy)
            for key in ("best_pick", "runner_up", "budget_pick"):
                pick = out.get(key)
                if not pick:
                    continue
                title_match = (pick.get("title") or "").strip()
                for p in valid_products:
                    if (p.get("title") or "").strip() == title_match and p.get("link"):
                        pick["link"] = p["link"]
                        if p.get("image_link"):
                            pick["image_link"] = p["image_link"]
                        break
            return out
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return None

    def process_shopping_list(self, json_input, raw_file="shopping_results_raw.json", final_file="shopping_results_final.json"):
        data = json_input if isinstance(json_input, dict) else json.loads(json_input)
        final_report = {}
        raw_results = {}

        pincode = data.get("delivery_pincode", "10001")
        deadline = data.get("delivery_deadline_date")
        total_budget = data.get("total_budget")
        budget_curr = data.get("budget_currency", "USD")
        location, local_curr, _ = get_location_from_pincode(pincode)
        items = data["items"]
        num_items = len(items)

        if total_budget:
            local_total = convert_budget_to_local(total_budget, budget_curr, local_curr)
        else:
            local_total = 100000.0 * num_items
            total_budget = "Unlimited"

        item_price_ranges = {}
        for item in items:
            query = item["query"]
            all_candidates = self.search_google_shopping(query, pincode)
            if all_candidates:
                prices = [p["price_val"] for p in all_candidates if p.get("price_val", 0) > 0]
                if prices:
                    item_price_ranges[query] = {
                        "min": min(prices), "max": max(prices),
                        "avg": sum(prices) / len(prices), "count": len(prices)
                    }

        total_avg = sum(r["avg"] for r in item_price_ranges.values()) if item_price_ranges else 0
        budget_allocation = {}
        if item_price_ranges:
            for query, pr in item_price_ranges.items():
                budget_allocation[query] = local_total * (pr["avg"] / total_avg)
        else:
            per_item = local_total / num_items
            budget_allocation = {item["query"]: per_item for item in items}

        total_cost_estimate = 0
        for idx, item in enumerate(items, 1):
            query = item["query"]
            per_item_budget = budget_allocation.get(query, local_total / num_items)
            all_candidates = self.search_google_shopping(query, pincode)
            unique_retailers = list(set([p.get("source", "Unknown") for p in all_candidates]))
            enriched = []
            for prod in all_candidates[:8]:
                p = dict(prod)
                self.enrich_product_link(p)
                enriched.append(p)
            raw_results[query] = {
                "all_products": all_candidates,
                "unique_retailers": unique_retailers,
                "total_retailers": len(unique_retailers),
                "total_products": len(all_candidates)
            }
            constraints = {
                "max_price": per_item_budget,
                "currency": local_curr,
                "deadline": deadline,
                "location": location,
                "preferences": item.get("preferences")
            }
            decision = self.rank_and_decide(query, enriched, constraints)
            best_pick_price = 0
            if decision and decision.get("best_pick"):
                try:
                    price_str = str(decision["best_pick"].get("price", "0"))
                    clean_price = re.sub(r"[^\d.]", "", price_str.replace(",", ""))
                    best_pick_price = float(clean_price) if clean_price else 0
                except (ValueError, TypeError):
                    pass
            total_cost_estimate += best_pick_price
            final_report[query] = {
                "recommendations": decision,
                "unique_retailers": unique_retailers,
                "total_retailers": len(unique_retailers),
                "total_products_searched": len(all_candidates),
                "allocated_budget": per_item_budget,
                "best_pick_price": best_pick_price,
                "budget_compliant": best_pick_price <= per_item_budget * 1.05,
                "savings": per_item_budget - best_pick_price
            }

        summary = {
            "budget_summary": {
                "total_budget_usd": total_budget,
                "total_budget_local": local_total,
                "currency": local_curr,
                "num_items": num_items,
                "allocation_method": "Dynamic (based on market prices)" if item_price_ranges else "Equal split",
                "price_ranges": item_price_ranges,
                "estimated_total_cost": total_cost_estimate,
                "budget_remaining": local_total - total_cost_estimate,
                "budget_compliant": total_cost_estimate <= local_total
            }
        }
        try:
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        final_report_with_summary = {**summary, **final_report}
        try:
            with open(final_file, "w", encoding="utf-8") as f:
                json.dump(final_report_with_summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return {"raw_results": raw_results, "final_results": final_report_with_summary}

# ==========================================
# FASTAPI ENDPOINTS
# ==========================================

def get_session(session_id):
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"messages": []}
    return session_id, sessions[session_id]


class ItemRequest(BaseModel):
    query: str
    preferences: Optional[str] = None


class ShoppingRequest(BaseModel):
    delivery_pincode: str
    delivery_deadline_date: Optional[str] = None
    total_budget: Optional[float] = None
    budget_currency: str = "USD"
    items: List[ItemRequest]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id, session = get_session(req.session_id)
    session["messages"].append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session["messages"]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.4,
    )
    assistant_reply = response.choices[0].message.content
    session["messages"].append({"role": "assistant", "content": assistant_reply})
    if "FINAL_JSON:" in assistant_reply:
        json_part = assistant_reply.split("FINAL_JSON:")[1].strip()
        try:
            parsed = json.loads(json_part)
            return ChatResponse(session_id=session_id, reply="Order specification completed.", json_output=parsed)
        except Exception:
            pass
    return ChatResponse(session_id=session_id, reply=assistant_reply)


@app.post("/shop")
async def shop_endpoint(request: ShoppingRequest):
    try:
        request_data = request.model_dump()
        agent = SmartShoppingAgent()
        result = agent.process_shopping_list(request_data)
        return result
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Shopping Assistant API...")
    print("   POST /chat - Chat with shopping assistant")
    print("   POST /shop - Search and recommend products")
    uvicorn.run(app, host="0.0.0.0", port=8000)