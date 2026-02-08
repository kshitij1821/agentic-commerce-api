import os
import re
import json
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import GoogleSearch

# --- Optional Imports with Safety Checks ---
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

# ==========================================
# 0. CONFIGURATION
# ==========================================
load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SERPAPI_KEY or not OPENAI_API_KEY:
    print("⚠️  WARNING: API Keys not found in .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Agentic Commerce API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Session Storage for Chat
sessions: Dict[str, Dict[str, Any]] = {}

# ==========================================
# 1. HELPER FUNCTIONS (Shopping Logic)
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
        except Exception:
            pass
    return ("United States", "USD", "us") if country == "us" else ("India", "INR", "in")

def convert_budget_to_local(amount: float, from_curr: str, to_curr: str) -> float:
    if from_curr == to_curr: return amount
    if CurrencyRates:
        try:
            c = CurrencyRates()
            return c.convert(from_curr, to_curr, amount)
        except Exception: pass
    
    rates = {"USD": 1.0, "INR": 84.0, "EUR": 0.92, "GBP": 0.79}
    try:
        usd_base = amount / rates.get(from_curr, 1.0)
        return round(usd_base * rates.get(to_curr, 1.0), 2)
    except:
        return amount

# ==========================================
# 2. SMART SHOPPING AGENT CLASS
# ==========================================
class SmartShoppingAgent:
    def __init__(self):
        self.serp_api_key = SERPAPI_KEY
        self.client = client

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
            local_total = 100000.0 * num_items # Default high budget

        # Budget Allocation Logic
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

        # Main Loop
        total_cost_estimate = 0
        for item in items:
            query = item["query"]
            per_item_budget = budget_allocation.get(query, local_total / num_items)
            
            # Search & Enrich
            all_candidates = self.search_google_shopping(query, pincode)
            enriched_candidates = []
            for prod in all_candidates[:5]:
                enriched_candidates.append(self.enrich_product_link(prod))
            
            # Store Raw
            raw_results[query] = {
                "all_products": all_candidates,
                "total_products": len(all_candidates)
            }
            
            # Decide
            constraints = {
                "max_price": per_item_budget,
                "currency": local_curr,
                "deadline": deadline,
                "location": location,
                "preferences": item.get("preferences")
            }
            decision = self.rank_and_decide(query, enriched_candidates, constraints)
            
            # Track Cost
            best_pick_price = 0
            if decision and 'best_pick' in decision:
                try:
                    price_str = str(decision['best_pick'].get('price', '0'))
                    clean_price = re.sub(r'[^\d.]', '', price_str.replace(',', ''))
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

# ==========================================
# 3. ENDPOINT DEFINITIONS
# ==========================================

# --- Models ---
class ItemRequest(BaseModel):
    query: str
    preferences: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str | None = None
    delivery_pincode: str = "400076"
    delivery_deadline_date: Optional[str] = None
    total_budget: Optional[float] = None
    budget_currency: str = "USD"
    items: List[ItemRequest]
    message: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    json_output: dict | None = None

class ShoppingRequest(BaseModel):
    delivery_pincode: str
    delivery_deadline_date: Optional[str] = None
    total_budget: Optional[float] = None
    budget_currency: str = "USD"
    items: List[ItemRequest]

class ShoppingResponse(BaseModel):
    raw_results: Dict[str, Any]
    final_results: Dict[str, Any]


# --- Chat Logic (Planner) ---
SYSTEM_PROMPT = """
You are an AI shopping assistant.

Your ONLY goal is to help users plan purchases.

Guardrails:
•⁠  ⁠Refuse unrelated requests politely.
•⁠  ⁠Keep responses concise (1–2 sentences).
•⁠  ⁠Ask only one question at a time.
•⁠  ⁠Guide users toward purchase planning.
•⁠  ⁠Do not engage in chit-chat.

Defaults:
•⁠  ⁠If delivery deadline is not specified, assume 14 days from today.
•⁠  ⁠If item preferences are not specified, use "No specific preferences".
•⁠  ⁠Delivery pincode defaults to 400076.
•⁠  ⁠Currency defaults to USD.

Conversation Flow:
1.⁠ ⁠Review the provided items, budget, and delivery details.
2.⁠ ⁠Ask clarifying questions if needed (e.g., additional preferences).
3.⁠ ⁠Summarize the order.
4.⁠ ⁠Ask for confirmation.
5.⁠ ⁠After confirmation, output FINAL_JSON with the provided data.

When user confirms, respond:

FINAL_JSON:
<json>

JSON format:

{
  "delivery_pincode": "...",
  "delivery_deadline_date": "...",
  "total_budget": number,
  "budget_currency": "...",
  "items": [
    {
      "query": "...",
      "preferences": "..."
    }
  ]
}

Rules:
•⁠  ⁠NEVER output FINAL_JSON before confirmation.
•⁠  ⁠Keep answers short.
•⁠  ⁠Stay within shopping assistance.
•⁠  ⁠Use the provided delivery_pincode, deadline, budget, and items.
•⁠  ⁠If deadline is not provided in request, set it to 14 days from today (today is 2026-02-08).
"""

def get_session(session_id):
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"messages": []}
    return session_id, sessions[session_id]

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id, session = get_session(req.session_id)
    
    # Format the structured request into a user message
    items_str = "; ".join([f"{item.query} (Preferences: {item.preferences or 'None'})" for item in req.items])
    deadline = req.delivery_deadline_date or (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
    user_message = f"""I want to order the following items:
- Items: {items_str}
- Delivery Pincode: {req.delivery_pincode}
- Delivery Deadline: {deadline}
- Total Budget: {req.total_budget or 'Not specified'} {req.budget_currency}
{f"Additional notes: {req.message}" if req.message else ""}"""
    
    session["messages"].append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session["messages"]

    response = client.chat.completions.create(
        model="gpt-4o-mini", # Cost effective model for chat
        messages=messages,
        temperature=0.4,
    )

    assistant_reply = response.choices[0].message.content
    session["messages"].append({"role": "assistant", "content": assistant_reply})

    # Detect final JSON
    json_output = None
    if "FINAL_JSON:" in assistant_reply:
        try:
            json_part = assistant_reply.split("FINAL_JSON:")[1].strip()
            # Clean up potential markdown code blocks
            if json_part.startswith("```json"):
                json_part = json_part.replace("```json", "").replace("```", "")
            parsed = json.loads(json_part)
            
            # Ensure the output has the correct structure with provided data
            json_output = {
                "delivery_pincode": parsed.get("delivery_pincode", req.delivery_pincode),
                "delivery_deadline_date": parsed.get("delivery_deadline_date", deadline),
                "total_budget": parsed.get("total_budget", req.total_budget),
                "budget_currency": parsed.get("budget_currency", req.budget_currency),
                "items": parsed.get("items", [item.dict() for item in req.items])
            }
            assistant_reply = "I've generated your shopping plan! Searching for the best deals now..."
        except:
            pass

    return ChatResponse(
        session_id=session_id,
        reply=assistant_reply,
        json_output=json_output
    )


# --- Shopping Logic (Executor) ---
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

@app.get("/")
def home():
    return {"status": "Agent is running 🚀", "endpoints": ["/chat", "/shop"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
