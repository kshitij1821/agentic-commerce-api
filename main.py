import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from serpapi import GoogleSearch
from openai import OpenAI

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

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SERPAPI_KEY or not OPENAI_API_KEY:
    print("⚠️  WARNING: API Keys not found in .env file.")

# ==========================================
# 1. HELPER FUNCTIONS (from v3.py)
# ==========================================

def _get_nomi(country):
    if pgeocode:
        try:
            return pgeocode.Nominatim(country)
        except Exception:
            pass
    return None

def _simplify_location(place: str, state: str, country_code: str) -> str:
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

def get_location_from_pincode(pincode: str) -> tuple[str, str, str]:
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
    except:
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
    except:
        pass
    return None

# ==========================================
# 2. MAIN AGENT CLASS (from v3.py, API-ready)
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
                    clean_p = re.sub(r'[^\\d.]', '', str(price_str).replace(',', ''))
                    try: price_val = float(clean_p)
                    except: pass
                clean_items.append({
                    "title": item.get("title"),
                    "price_str": price_str,
                    "price_val": price_val,
                    "currency": currency,
                    "source": item.get("source", "Unknown"),
                    "link": item.get("link"),
                    "delivery": item.get("delivery", "Unknown"),
                    "thumbnail": item.get("thumbnail"),
                    "rating": item.get("rating", "N/A"),
                    "reviews": item.get("reviews", 0)
                })
            return clean_items
        except Exception as e:
            print(f"❌ Search Error: {e}")
            return []

    def enrich_product_link(self, product):
        original_link = product.get("link") or ""
        source = (product.get("source") or "").lower()
        title = product.get("title") or ""
        needs_fix = not original_link or "google.com" in original_link or "/aclk" in original_link
        if needs_fix and title:
            search_query = f"{title}"
            if "decathlon" in source: search_query += " site:decathlon.in"
            elif "amazon" in source: search_query += " site:amazon.in"
            elif "flipkart" in source: search_query += " site:flipkart.com"
            elif "myntra" in source: search_query += " site:myntra.com"
            else: search_query += f" site:{source.replace(' ', '').lower()}.com"
            params = {"api_key": self.serp_api_key, "engine": "google", "q": search_query, "num": 1}
            try:
                search = GoogleSearch(params)
                res = search.get_dict()
                organic = res.get("organic_results", [])
                if organic:
                    product["link"] = organic[0].get("link")
                    snippet = organic[0].get("snippet", "")
                    if "delivery" not in str(product.get("delivery")).lower():
                        product["delivery"] = f"Check Site ({snippet[:60]}...)"
            except Exception: pass
        return product

    def rank_and_decide(self, query, products, constraints):
        max_price = constraints.get("max_price", float('inf'))
        currency = constraints.get("currency", "USD")
        valid_products = [p for p in products if p['price_val'] <= max_price]
        if not valid_products:
            valid_products = sorted(products, key=lambda x: x['price_val'])[:5]
        retailers = list(set([p.get('source', 'Unknown') for p in valid_products]))
        retailers_str = ", ".join(retailers[:10])
        prompt = f"""
        You are a shopping assistant with STRICT budget enforcement.
        GOAL: Select the best 3 options for \"{query}\" from DIFFERENT RETAILERS within budget.
        **CRITICAL BUDGET CONSTRAINT:** ₹{max_price} {currency} per item.
        **RETAILER DIVERSITY:** All 3 picks MUST be from DIFFERENT retailers. Available: {retailers_str}
        CONTEXT:
        - Location: {constraints.get('location')}
        - Budget: {max_price} {currency}
        - Deadline: {constraints.get('deadline')}
        - Preferences: {constraints.get('preferences')}
        PRODUCTS:
        {json.dumps(valid_products, indent=2)}
        OUTPUT JSON format:
        {{
            "best_pick": {{ "title": "...", "price": "...", "source": "...", "link": "...", "image_link": "...", "delivery": "...", "reasoning": "..." }},
            "runner_up": {{ "title": "...", "price": "...", "source": "...", "link": "...", "image_link": "...", "delivery": "...", "reasoning": "..." }},
            "budget_pick": {{ "title": "...", "price": "...", "source": "...", "link": "...", "image_link": "...", "delivery": "...", "reasoning": "..." }}
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "Return valid JSON only."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return None

    def process_shopping_list(self, data: Dict[str, Any]):
        final_report = {}
        raw_results = {}
        pincode = data.get("delivery_pincode", "10001")
        deadline = data.get("delivery_deadline_date")
        total_budget = data.get("total_budget")
        budget_curr = data.get("budget_currency", "USD")
        location, local_curr, gl = get_location_from_pincode(pincode)
        items = data["items"]
        num_items = len(items)
        if total_budget:
            local_total = convert_budget_to_local(total_budget, budget_curr, local_curr)
        else:
            local_total = 100000.0 * num_items
        item_price_ranges = {}
        for item in items:
            query = item["query"]
            all_candidates = self.search_google_shopping(query, pincode)
            if all_candidates:
                prices = [p['price_val'] for p in all_candidates if p['price_val'] > 0]
                if prices:
                    item_price_ranges[query] = {"avg": sum(prices) / len(prices)}
        budget_allocation = {}
        if item_price_ranges:
            total_avg = sum(r['avg'] for r in item_price_ranges.values())
            for query, r in item_price_ranges.items():
                budget_allocation[query] = local_total * (r['avg'] / total_avg)
        else:
            for item in items:
                budget_allocation[item["query"]] = local_total / num_items
        total_cost_estimate = 0
        for item in items:
            query = item["query"]
            per_item_budget = budget_allocation.get(query, local_total / num_items)
            all_candidates = self.search_google_shopping(query, pincode)
            enriched_candidates = []
            for prod in all_candidates[:5]:
                enriched_candidates.append(self.enrich_product_link(prod))
            raw_results[query] = {
                "all_products": all_candidates,
                "total_products": len(all_candidates)
            }
            constraints = {
                "max_price": per_item_budget,
                "currency": local_curr,
                "deadline": deadline,
                "location": location,
                "preferences": item.get("preferences")
            }
            decision = self.rank_and_decide(query, enriched_candidates, constraints)
            best_pick_price = 0
            if decision and 'best_pick' in decision:
                try:
                    price_str = str(decision['best_pick'].get('price', '0'))
                    clean_price = re.sub(r'[^\\\d.]', '', price_str.replace(',', ''))
                    best_pick_price = float(clean_price) if clean_price else 0
                except: pass
            total_cost_estimate += best_pick_price
            final_report[query] = {
                "recommendations": decision,
                "allocated_budget": per_item_budget,
                "best_pick_price": best_pick_price
            }
        summary = {
            "total_budget_local": local_total,
            "currency": local_curr,
            "estimated_total_cost": total_cost_estimate,
            "budget_compliant": total_cost_estimate <= local_total
        }
        return {
            "raw_results": raw_results,
            "final_results": {**summary, "items": final_report}
        }

    def chat(self, message: str):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": message}],
                response_format={"type": "text"}
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Chat error: {e}"

# ==========================================
# 3. FASTAPI APP SETUP (combined)
# ==========================================

app = FastAPI(title="Agentic Commerce API (Combined)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ItemRequest(BaseModel):
    query: str
    preferences: Optional[str] = None

class ShoppingRequest(BaseModel):
    delivery_pincode: str
    delivery_deadline_date: Optional[str] = None
    total_budget: Optional[float] = None
    budget_currency: str = "USD"
    items: List[ItemRequest]

class ShoppingResponse(BaseModel):
    raw_results: Dict[str, Any]
    final_results: Dict[str, Any]

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/shop", response_model=ShoppingResponse)
async def shop_endpoint(request: ShoppingRequest):
    try:
        request_data = request.dict()
        agent = SmartShoppingAgent()
        result = agent.process_shopping_list(request_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        agent = SmartShoppingAgent()
        response = agent.chat(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"status": "Agent is running 🚀"}

# ==========================================
# 4. RUNNER (For Local Testing)
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
