```markdown
# accounts.py – Detailed Design Document

Module name: `accounts.py`  
Primary class: `Account`  
Dependencies: none (self-contained; includes a stub for `get_share_price`)  

Purpose  
Provide an in-memory, single-file trading-simulation ledger that supports cash & equity operations, real-time valuation, P&L tracking, and immutable transaction history with defensive validations.

---

## 1. Module-level helpers

| Name | Signature | Description |
|---|---|---|
| `get_share_price` | `(symbol: str) -> Decimal` | Returns current market price.  Built-in stub maps AAPL→150, TSLA→800, GOOGL→120; everything else →100. |

---

## 2. Data Transfer Objects (internal, pure dataclasses)

| Name | Fields | Purpose |
|---|---|---|
| `Transaction` | `t_id: str` – UUID4 hex string <br> `ts: datetime` – UTC timestamp <br> `symbol: str \| None` – None for cash tx <br> `quantity: Decimal` – +ve buy / –ve sell <br> `price: Decimal \| None` – None for cash tx <br> `cash_delta: Decimal` – +ve deposit / –ve withdraw <br> `note: str` – human readable | Immutable row in the ledger |

---

## 3. Main class – Account

| Member | Type | Description |
|---|---|---|
| `_cash` | `Decimal` | Current cash balance (2 dp) |
| `_holdings` | `dict[str, Decimal]` | symbol → quantity mapping |
| `_history` | `list[Transaction]` | Chronological, append-only |
| `_initial_deposit` | `Decimal` | Seed money for lifetime P&L |

---

### 3.1 Public API

| Method | Signature | Functional spec & validations |
|---|---|---|
| `__init__` | `(owner: str = "")` | Creates account with zero cash & empty portfolio.  `owner` stored for display only. |
| `deposit` | `(amount: Decimal \| int \| float) -> Transaction` | Adds cash.  Rejects non-positive amounts.  Returns transaction object. |
| `withdraw` | `(amount: Decimal \| int \| float) -> Transaction` | Removes cash.  Reject if `amount <= 0` or would make `_cash < 0`.  Returns transaction object. |
| `buy` | `(symbol: str, quantity: Decimal \| int \| float) -> Transaction` | Market-order purchase.  Reject if `quantity <= 0` or `cash < quantity * get_share_price(symbol)`.  Updates `_cash` & `_holdings`.  Returns transaction object. |
| `sell` | `(symbol: str, quantity: Decimal \| int \| float) -> Transaction` | Market-order sale.  Reject if `quantity <= 0` or `_holdings.get(symbol,0) < quantity`.  Updates `_cash` & `_holdings`.  Returns transaction object. |
| `get_cash_balance` | `() -> Decimal` | Current cash. |
| `get_holdings` | `() -> dict[str, Decimal]` | Defensive copy of share map. |
| `get_portfolio_value` | `() -> Decimal` | Sum of `holdings[k] * get_share_price(k)` + cash. |
| `get_lifetime_pnl` | `() -> Decimal` | `portfolio_value - _initial_deposit`.  Negative if underwater. |
| `get_transactions` | `(limit: int \| None = None) -> list[Transaction]` | Chronological list (newest last).  Slice with `limit` if provided. |
| `get_symbol_pnl` | `(symbol: str) -> Decimal` | Realised P&L for one symbol: aggregate cash deltas from sell transactions minus buy deltas for that symbol. |

---

### 3.2 Private helpers

| Method | Signature | Description |
|---|---|---|
| `_validate_decimal` | `(value, name) -> Decimal` | Coerce & assert > 0 for amounts/quantities. |
| `_new_tx` | `(...) -> Transaction` | Factory that appends to `_history` and returns instance. |

---

## 4. Example usage (will not be shipped inside module)

```python
from decimal import Decimal
from accounts import Account

acc = Account("Alice")
acc.deposit(10_000)
acc.buy("AAPL", 10)          # spends 1 500
acc.sell("AAPL", 5)          # receives 750
print(acc.get_portfolio_value())
print(acc.get_lifetime_pnl())
for tx in acc.get_transactions():
    print(tx)
```

---

## 5. File layout (single self-contained module)

```
accounts.py
├─ Imports (datetime, uuid, Decimal)
├─ get_share_price (stub)
├─ Transaction (dataclass)
└─ Account (class with all above members)
```

The module is ready for `pytest` or a thin CLI/UI wrapper.
```