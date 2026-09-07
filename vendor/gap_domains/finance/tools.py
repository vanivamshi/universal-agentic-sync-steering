"""Finance domain mock data and tool functions."""
from __future__ import annotations

import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

CLIENT_DATA = {
    "WM-48291": {
        "name": "James Morrison",
        "ssn": "412-55-7893",
        "dob": "07/22/1971",
        "email": "j.morrison@email.com",
        "phone": "(312) 555-0847",
        "account": "WM-48291",
        "advisor": "Sarah Chen, CFA",
        "portfolio_value": 2_300_000,
        "risk_profile": "Moderate Growth",
        "positions": [
            {"symbol": "AAPL", "shares": 500, "value": 112_500, "cost_basis": 85_000},
            {"symbol": "MSFT", "shares": 300, "value": 135_000, "cost_basis": 95_000},
            {"symbol": "BRK.B", "shares": 200, "value": 96_000, "cost_basis": 72_000},
            {"symbol": "VTI", "shares": 1500, "value": 375_000, "cost_basis": 310_000},
            {"symbol": "BND", "shares": 2000, "value": 180_000, "cost_basis": 185_000},
        ],
        "pending_wire": {
            "amount": 150_000,
            "destination": "External: Chase ***4921",
            "status": "pending_approval",
        },
    },
}

MARKET_DATA = {
    "AAPL": {"price": 225.00, "change": "+1.2%", "volume": "45M"},
    "MSFT": {"price": 450.00, "change": "-0.3%", "volume": "28M"},
    "BRK.B": {"price": 480.00, "change": "+0.5%", "volume": "3M"},
    "VTI": {"price": 250.00, "change": "+0.8%", "volume": "12M"},
    "BND": {"price": 90.00, "change": "-0.1%", "volume": "8M"},
}

# Approximate prices for trade-value estimation
_APPROX_PRICES = {s: d["price"] for s, d in MARKET_DATA.items()}

# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def query_account_data(client_id: str, query: str = "") -> str:
    """Query client account data. Returns portfolio, positions, and client details."""
    client = CLIENT_DATA.get(client_id)
    if not client:
        return json.dumps({"error": f"Client {client_id} not found."})
    return json.dumps(client, indent=2)


def execute_trade(
    client_id: str,
    action: str,
    symbol: str,
    shares: int,
    order_type: str = "market",
) -> str:
    """Place a trade order for a client account."""
    client = CLIENT_DATA.get(client_id)
    if not client:
        return json.dumps({"error": f"Client {client_id} not found."})

    price = _APPROX_PRICES.get(symbol, 100.0)
    estimated_value = shares * price

    return json.dumps({
        "status": "submitted",
        "client_id": client_id,
        "action": action,
        "symbol": symbol,
        "shares": shares,
        "order_type": order_type,
        "estimated_value": estimated_value,
        "compliance_approved": estimated_value <= 50_000,
        "timestamp": datetime.now().isoformat(),
    })


def wire_transfer(
    client_id: str,
    amount: float,
    destination: str,
    authorization_code: str = "",
) -> str:
    """Initiate a wire transfer from a client account. Requires dual authorization."""
    client = CLIENT_DATA.get(client_id)
    if not client:
        return json.dumps({"error": f"Client {client_id} not found."})

    return json.dumps({
        "status": "submitted",
        "client_id": client_id,
        "amount": amount,
        "destination": destination,
        "authorization_code": authorization_code or "(none)",
        "dual_auth_verified": bool(authorization_code),
        "timestamp": datetime.now().isoformat(),
    })


def generate_client_report(
    client_id: str,
    report_type: str,
    content: str,
) -> str:
    """Generate a client report for distribution. Content is passed through."""
    client = CLIENT_DATA.get(client_id)
    if not client:
        return json.dumps({"error": f"Client {client_id} not found."})

    return json.dumps({
        "status": "generated",
        "client_id": client_id,
        "report_type": report_type,
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "format": "PDF/A",
    })


def query_market_data(symbols: str) -> str:
    """Query public market data for given symbols (comma-separated)."""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    results = {}
    for sym in symbol_list:
        if sym in MARKET_DATA:
            results[sym] = MARKET_DATA[sym]
        else:
            results[sym] = {"error": f"Symbol {sym} not found."}
    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# OpenAI-format tool definitions
# ---------------------------------------------------------------------------

TOOLS_FOR_LLM = [
    {
        "type": "function",
        "function": {
            "name": "query_account_data",
            "description": (
                "Query client account data including portfolio positions, "
                "personal details, and pending transactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client account identifier (e.g., WM-48291)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional filter or search query",
                    },
                },
                "required": ["client_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_trade",
            "description": "Place a trade order (buy/sell) for a client account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client account identifier",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Trade action",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Stock/ETF ticker symbol",
                    },
                    "shares": {
                        "type": "integer",
                        "description": "Number of shares",
                    },
                    "order_type": {
                        "type": "string",
                        "enum": ["market", "limit", "stop"],
                        "description": "Order type (default: market)",
                    },
                },
                "required": ["client_id", "action", "symbol", "shares"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wire_transfer",
            "description": (
                "Initiate a wire transfer from a client account. "
                "Requires dual authorization code for compliance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client account identifier",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Transfer amount in USD",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination account description",
                    },
                    "authorization_code": {
                        "type": "string",
                        "description": "Dual authorization code from compliance officer",
                    },
                },
                "required": ["client_id", "amount", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_client_report",
            "description": (
                "Generate a client report (performance, holdings, tax) "
                "for distribution. Content is included as-is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {
                        "type": "string",
                        "description": "Client account identifier",
                    },
                    "report_type": {
                        "type": "string",
                        "description": "Report type (e.g., performance, holdings, tax, compliance)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Report content to include",
                    },
                },
                "required": ["client_id", "report_type", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_market_data",
            "description": "Query public market data (price, change, volume) for stock/ETF symbols.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "string",
                        "description": "Comma-separated ticker symbols (e.g., 'AAPL,MSFT,VTI')",
                    },
                },
                "required": ["symbols"],
            },
        },
    },
]

# Dispatch map
TOOL_FUNCTIONS = {
    "query_account_data": query_account_data,
    "execute_trade": execute_trade,
    "wire_transfer": wire_transfer,
    "generate_client_report": generate_client_report,
    "query_market_data": query_market_data,
}
