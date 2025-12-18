from decimal import Decimal, getcontext
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Dict, List
import uuid

def get_share_price(symbol: str) -> Decimal:
    """Returns current market price. Built-in stub maps AAPL→150, TSLA→800, GOOGL→120; everything else →100."""
    prices = {
        'AAPL': Decimal('150'),
        'TSLA': Decimal('800'),
        'GOOGL': Decimal('120')
    }
    return prices.get(symbol, Decimal('100'))

@dataclass(frozen=True)
class Transaction:
    """Immutable row in the ledger"""
    t_id: str
    ts: datetime
    symbol: Optional[str]
    quantity: Decimal
    price: Optional[Decimal]
    cash_delta: Decimal
    note: str

class Account:
    """Account management system for trading simulation"""
    
    def __init__(self, owner: str = ""):
        """Creates account with zero cash & empty portfolio"""
        self._owner = owner
        self._cash = Decimal('0')
        self._holdings: Dict[str, Decimal] = {}
        self._history: List[Transaction] = []
        self._initial_deposit = Decimal('0')
        
    def _validate_decimal(self, value, name: str) -> Decimal:
        """Coerce & assert > 0 for amounts/quantities"""
        if isinstance(value, (int, float)):
            value = Decimal(str(value))
        if not isinstance(value, Decimal):
            raise TypeError(f"{name} must be a Decimal, int, or float")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value.quantize(Decimal('0.01'))
    
    def _new_tx(self, symbol: Optional[str], quantity: Decimal, price: Optional[Decimal], 
                cash_delta: Decimal, note: str) -> Transaction:
        """Factory that appends to _history and returns instance"""
        tx = Transaction(
            t_id=uuid.uuid4().hex,
            ts=datetime.now(timezone.utc),
            symbol=symbol,
            quantity=quantity,
            price=price,
            cash_delta=cash_delta,
            note=note
        )
        self._history.append(tx)
        return tx
    
    def deposit(self, amount) -> Transaction:
        """Adds cash. Rejects non-positive amounts."""
        amount = self._validate_decimal(amount, "amount")
        self._cash += amount
        # Set initial deposit only for the first deposit
        if self._initial_deposit == Decimal('0'):
            self._initial_deposit = amount
        return self._new_tx(None, Decimal('0'), None, amount, f"Deposit ${amount}")
    
    def withdraw(self, amount) -> Transaction:
        """Removes cash. Reject if amount <= 0 or would make _cash < 0."""
        amount = self._validate_decimal(amount, "amount")
        if amount > self._cash:
            raise ValueError("Insufficient funds for withdrawal")
        self._cash -= amount
        return self._new_tx(None, Decimal('0'), None, -amount, f"Withdraw ${amount}")
    
    def buy(self, symbol: str, quantity) -> Transaction:
        """Market-order purchase. Reject if quantity <= 0 or cash < quantity * price."""
        quantity = self._validate_decimal(quantity, "quantity")
        price = get_share_price(symbol)
        cost = (quantity * price).quantize(Decimal('0.01'))
        
        if cost > self._cash:
            raise ValueError(f"Insufficient funds to buy {quantity} shares of {symbol}")
            
        self._cash -= cost
        self._holdings[symbol] = self._holdings.get(symbol, Decimal('0')) + quantity
        
        return self._new_tx(symbol, quantity, price, -cost, f"Buy {quantity} shares of {symbol} at ${price} per share")
    
    def sell(self, symbol: str, quantity) -> Transaction:
        """Market-order sale. Reject if quantity <= 0 or holdings < quantity."""
        quantity = self._validate_decimal(quantity, "quantity")
        holding = self._holdings.get(symbol, Decimal('0'))
        
        if quantity > holding:
            raise ValueError(f"Cannot sell {quantity} shares of {symbol}; only {holding} owned")
            
        price = get_share_price(symbol)
        proceeds = (quantity * price).quantize(Decimal('0.01'))
        
        self._cash += proceeds
        self._holdings[symbol] -= quantity
        # Remove symbol from holdings if quantity is zero
        if self._holdings[symbol] == Decimal('0'):
            del self._holdings[symbol]
            
        return self._new_tx(symbol, -quantity, price, proceeds, f"Sell {quantity} shares of {symbol} at ${price} per share")
    
    def get_cash_balance(self) -> Decimal:
        """Current cash."""
        return self._cash.quantize(Decimal('0.01'))
    
    def get_holdings(self) -> Dict[str, Decimal]:
        """Defensive copy of share map."""
        return self._holdings.copy()
    
    def get_portfolio_value(self) -> Decimal:
        """Sum of holdings[k] * get_share_price(k) + cash."""
        holdings_value = sum(
            quantity * get_share_price(symbol)
            for symbol, quantity in self._holdings.items()
        )
        return (self._cash + holdings_value).quantize(Decimal('0.01'))
    
    def get_lifetime_pnl(self) -> Decimal:
        """portfolio_value - _initial_deposit. Negative if underwater."""
        if self._initial_deposit == Decimal('0'):
            return Decimal('0')
        return (self.get_portfolio_value() - self._initial_deposit).quantize(Decimal('0.01'))
    
    def get_transactions(self, limit: Optional[int] = None) -> List[Transaction]:
        """Chronological list (newest last). Slice with limit if provided."""
        if limit is None:
            return self._history.copy()
        return self._history[-limit:].copy()
    
    def get_symbol_pnl(self, symbol: str) -> Decimal:
        """Realised P&L for one symbol: aggregate cash deltas from sell transactions minus buy deltas for that symbol."""
        symbol_pnl = Decimal('0')
        for tx in self._history:
            if tx.symbol == symbol:
                # For buys, cash_delta is negative (money spent)
                # For sells, cash_delta is positive (money received)
                symbol_pnl += tx.cash_delta
        return symbol_pnl.quantize(Decimal('0.01'))