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

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


If the user asks for general categories, you MUST break it down into specific searchable cattegories.
Example - "electronics"-> "laptop", "mouse", "keyboard"



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
- NEVER leave fields null; apply defaults if needed 
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

def get_location_from_pincode(pincode: str) -> tuple[str, str, str]:
    """Maps pincode -> (Location String, Currency, Country Code)."""
    pincode = str(pincode).strip()
    # Simple heuristic: 6 digits = India, 5 digits = US
    country = "in" if len(pincode) == 6 and pincode.isdigit() else "us"
    nomi = _get_nomi(country)

    if nomi:
        try:
            result = nomi.query_postal_code(pincode)
            if result is not None:
                place = getattr(result, "place_name", None)
                state = getattr(result, "state_name", None)
                
                # Handle pandas series or dict-like objects
                if hasattr(result, 'values'): 
                    # pgeocode often returns a pandas object, safer to convert to string if needed
                    place = str(place) if place and str(place) != 'nan' else ""
                    state = str(state) if state and str(state) != 'nan' else ""
                
                if place or state:
                    loc = _simplify_location(place, state, country)
                    currency = "INR" if country == "in" else "USD"
                    return (loc, currency, country)
        except Exception as e:
            print(f"⚠️ Location lookup warning: {e}")

    # Fallbacks
    return ("United States", "USD", "us") if country == "us" else ("India", "INR", "in")

def convert_budget_to_local(amount: float, from_curr: str, to_curr: str) -> float:
    """Converts budget (e.g. $100 -> ₹8300)."""
    if from_curr == to_curr:
        return amount
    
    # Try Forex Python
    if CurrencyRates:
        try:
            c = CurrencyRates()
            return c.convert(from_curr, to_curr, amount)
        except Exception:
            pass
    
    # Hardcoded Fallbacks
    rates = {"USD": 1.0, "INR": 84.0, "EUR": 0.92, "GBP": 0.79}
    try:
        usd_base = amount / rates.get(from_curr, 1.0)
        return round(usd_base * rates.get(to_curr, 1.0), 2)
    except:
        return amount

def parse_delivery_date(delivery_str: str, base_date: datetime):
    """Parses 'Free delivery by Wed, Feb 14' into '2026-02-14'."""
    if not delivery_str or "unknown" in str(delivery_str).lower():
        return None
    
    s = str(delivery_str).lower()
    
    # Relative terms
    if "tomorrow" in s:
        return (base_date + timedelta(days=1)).strftime("%Y-%m-%d")
    if "today" in s:
        return base_date.strftime("%Y-%m-%d")
    
    # Regex for dates
    try:
        # Match "Feb 14" or "Oct 22"
        match = re.search(r"([a-z]{3})\s(\d{1,2})", s)
        if match:
            # Assume current year, handle rollover later if needed
            dt = date_parser.parse(f"{match.group(1)} {match.group(2)} {base_date.year}")
            # If date is in the past (e.g. Dec search for Jan), add year
            if dt < base_date - timedelta(days=30):
                dt = dt.replace(year=base_date.year + 1)
            return dt.strftime("%Y-%m-%d")
            
        # Fallback to fuzzy parsing
        if date_parser:
            dt = date_parser.parse(s, fuzzy=True, default=base_date)
            return dt.strftime("%Y-%m-%d")
    except:
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
            # Fallback to cheapest if none fit budget
            valid_products = sorted(products, key=lambda x: x['price_val'])[:5]

        retailers = list(set([p.get('source', 'Unknown') for p in valid_products]))
        retailers_str = ", ".join(retailers[:10])

        prompt = f"""
        You are a shopping assistant with STRICT budget enforcement.
        GOAL: Select the best 3 options for "{query}" from DIFFERENT RETAILERS within budget.
        make the reasoning comparative (e.g. "Best Pick: X because it has free delivery and lowest price. Runner-up: Y is slightly more expensive but has better ratings. Budget Pick: Z is the cheapest option but has longer delivery time.")
        
        **CRITICAL BUDGET CONSTRAINT:** {max_price} {currency} per item.
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

    def process_shopping_list(self, json_input, raw_file="shopping_results_raw.json", final_file="shopping_results_final.json"):
        data = json_input if isinstance(json_input, dict) else json.loads(json_input)
        final_report = {}
        raw_results = {}
        
        # 1. Global Context
        pincode = data.get("delivery_pincode", "10001")
        deadline = data.get("delivery_deadline_date")
        total_budget = data.get("total_budget") # e.g. 200 USD
        budget_curr = data.get("budget_currency", "USD") # e.g. USD
        
        location, local_curr, gl = get_location_from_pincode(pincode)
        
        # Initial variables
        items = data["items"]
        num_items = len(items)
        
        if total_budget:
            local_total = convert_budget_to_local(total_budget, budget_curr, local_curr)
        else:
            local_total = 100000.0 * num_items
            total_budget = "Unlimited"

        print(f"\n{'='*60}")
        print(f"📍 Location: {location} | Currency: {local_curr}")
        print(f"💰 TOTAL BUDGET: {total_budget} {budget_curr} = {local_total:,.0f} {local_curr}")
        print(f"{'='*60}\n")

        # ========== PHASE 1: DISCOVERY - Quick scan to understand price ranges ==========
        print("📊 PHASE 1: Scanning market prices for each item...")
        item_price_ranges = {}
        
        for item in items:
            query = item["query"]
            all_candidates = self.search_google_shopping(query, pincode)
            
            if all_candidates:
                prices = [p['price_val'] for p in all_candidates if p['price_val'] > 0]
                if prices:
                    min_price = min(prices)
                    max_price = max(prices)
                    avg_price = sum(prices) / len(prices)
                    item_price_ranges[query] = {
                        "min": min_price,
                        "max": max_price,
                        "avg": avg_price,
                        "count": len(prices)
                    }
                    print(f"   {query}: ₹{min_price:,.0f} - ₹{max_price:,.0f} (avg: ₹{avg_price:,.0f})")
                    
        # ========== PHASE 2: Dynamic Budget Allocation ==========
        print(f"\n📈 PHASE 2: Calculating optimal budget allocation...")
        
        if item_price_ranges:
            # Calculate allocation based on average prices
            total_avg = sum(r['avg'] for r in item_price_ranges.values())
            budget_allocation = {}
            
            for query, price_range in item_price_ranges.items():
                # Allocate proportionally based on average price
                proportion = price_range['avg'] / total_avg
                allocated = local_total * proportion
                budget_allocation[query] = allocated
                print(f"   {query}: ₹{allocated:,.0f} ({proportion*100:.1f}% of budget)")
        else:
            # Fallback to equal allocation
            per_item = local_total / num_items
            budget_allocation = {item["query"]: per_item for item in items}
            print(f"   Using equal allocation: ₹{per_item:,.0f} per item")

        print(f"\n{'='*60}")
        print(f"📋 BUDGET ALLOCATION SUMMARY:")
        for query, budget in budget_allocation.items():
            print(f"   {query}: ₹{budget:,.0f}")
        print(f"{'='*60}\n")

        # ========== PHASE 3: Detailed Search & Ranking with Optimized Budgets ==========
        print("🔍 PHASE 3: Detailed search with optimized budgets...\n")
        
        all_selections = {}
        total_cost_estimate = 0
        
        for idx, item in enumerate(items, 1):
            query = item["query"]
            per_item_budget = budget_allocation.get(query, local_total / num_items)
            
            print(f"[Item {idx}/{num_items}] 🛍️  {query}")
            print(f"   Budget: ₹{per_item_budget:,.0f}")
            
            # A. Search (Get ALL results)
            all_candidates = self.search_google_shopping(query, pincode)
            
            # B. Extract unique retailers from ALL results
            unique_retailers = list(set([p.get('source', 'Unknown') for p in all_candidates]))
            num_unique_retailers = len(unique_retailers)
            
            # C. Enrich Top 5 for final ranking (but keep all in raw)
            enriched_candidates = []
            for prod in all_candidates[:5]:
                enriched_candidates.append(self.enrich_product_link(prod))
            
            # D. Store ALL results in raw_results with metadata
            raw_results[query] = {
                "all_products": all_candidates,
                "unique_retailers": unique_retailers,
                "total_retailers": num_unique_retailers,
                "total_products": len(all_candidates)
            }
            
            # E. Rank (using enriched top 5)
            constraints = {
                "max_price": per_item_budget,
                "currency": local_curr,
                "deadline": deadline,
                "location": location,
                "preferences": item.get("preferences")
            }
            
            decision = self.rank_and_decide(query, enriched_candidates, constraints)
            
            # F. Extract best pick price for budget tracking
            best_pick_price = 0
            if decision and 'best_pick' in decision:
                try:
                    price_str = str(decision['best_pick'].get('price', '0'))
                    clean_price = re.sub(r'[^\d.]', '', price_str.replace(',', ''))
                    best_pick_price = float(clean_price) if clean_price else 0
                except:
                    pass
            
            total_cost_estimate += best_pick_price
            all_selections[query] = decision
            
            # G. Add metadata to final results
            final_report[query] = {
                "recommendations": decision,
                "unique_retailers": unique_retailers,
                "total_retailers": num_unique_retailers,
                "total_products_searched": len(all_candidates),
                "allocated_budget": per_item_budget,
                "best_pick_price": best_pick_price,
                "budget_compliant": best_pick_price <= per_item_budget * 1.05,  # Allow 5% margin
                "savings": per_item_budget - best_pick_price
            }

        # 3. Add Summary & Total Budget Check
        summary = {
            "budget_summary": {
                "total_budget_usd": total_budget if isinstance(total_budget, str) else total_budget,
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
        
        print(f"\n{'='*60}")
        print(f"💸 FINAL BUDGET COMPLIANCE CHECK:")
        print(f"   Total Budget: {local_total:,.0f} {local_curr}")
        print(f"   Estimated Cost (Best Picks): {total_cost_estimate:,.0f} {local_curr}")
        print(f"   Remaining: {local_total - total_cost_estimate:,.0f} {local_curr}")
        print(f"   ✓ COMPLIANT: {total_cost_estimate <= local_total}")
        print(f"{'='*60}\n")

        # 4. Save Both Files
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Raw results (all searches + metadata) saved to: {raw_file}")
        
        # Add summary to final results
        final_report_with_summary = {**summary, **final_report}
        
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(final_report_with_summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Final results (top 3 from different retailers) saved to: {final_file}")
        
        return final_report_with_summary

# ==========================================
# FASTAPI ENDPOINTS
# ==========================================

def get_session(session_id):
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "messages": []
        }
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

class ShoppingResponse(BaseModel):
    raw_results: Dict[str, Any]
    final_results: Dict[str, Any]
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint that handles conversation with the shopping assistant.
    """
    session_id, session = get_session(req.session_id)

    session["messages"].append(
        {"role": "user", "content": req.message}
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += session["messages"]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.4,
    )

    assistant_reply = response.choices[0].message.content
    session["messages"].append(
        {"role": "assistant", "content": assistant_reply}
    )

    # Detect final JSON
    if "FINAL_JSON:" in assistant_reply:
        json_part = assistant_reply.split("FINAL_JSON:")[1].strip()
        try:
            parsed = json.loads(json_part)
            return ChatResponse(
                session_id=session_id,
                reply="Order specification completed.",
                json_output=parsed,
            )
        except:
            pass

    return ChatResponse(
        session_id=session_id,
        reply=assistant_reply,
    )

@app.post("/shop", response_model=ShoppingResponse)
async def shop_endpoint(request: ShoppingRequest):
    """
    Takes a structured shopping plan (likely from /chat), runs the Agent, 
    and returns both raw and final results.
    """
    try:
        request_data = request.dict()
        agent = SmartShoppingAgent()
        result = agent.process_shopping_list(request_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Shopping Assistant API...")
    print("📡 Endpoints available:")
    print("   POST /chat - Chat with shopping assistant")
    print("   POST /search - Search for products directly")
    print("\n📝 To test, run: python test_client.py --mode interactive")
    uvicorn.run(app, host="0.0.0.0", port=8000)