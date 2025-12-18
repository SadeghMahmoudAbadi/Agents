import unittest
from decimal import Decimal
from datetime import datetime, timezone
from accounts import Account, Transaction, get_share_price

class TestGetSharePrice(unittest.TestCase):
    def test_known_symbols(self):
        self.assertEqual(get_share_price('AAPL'), Decimal('150'))
        self.assertEqual(get_share_price('TSLA'), Decimal('800'))
        self.assertEqual(get_share_price('GOOGL'), Decimal('120'))
    
    def test_unknown_symbol(self):
        self.assertEqual(get_share_price('UNKNOWN'), Decimal('100'))

class TestTransaction(unittest.TestCase):
    def test_creation(self):
        tx = Transaction(
            t_id='test_id',
            ts=datetime.now(timezone.utc),
            symbol='AAPL',
            quantity=Decimal('10'),
            price=Decimal('150'),
            cash_delta=Decimal('-1500'),
            note='Test transaction'
        )
        self.assertEqual(tx.t_id, 'test_id')
        self.assertEqual(tx.symbol, 'AAPL')
        self.assertEqual(tx.quantity, Decimal('10'))
        self.assertEqual(tx.price, Decimal('150'))
        self.assertEqual(tx.cash_delta, Decimal('-1500'))
        self.assertEqual(tx.note, 'Test transaction')

class TestAccount(unittest.TestCase):
    def setUp(self):
        self.account = Account('Test Owner')
    
    def test_initial_state(self):
        self.assertEqual(self.account._owner, 'Test Owner')
        self.assertEqual(self.account.get_cash_balance(), Decimal('0'))
        self.assertEqual(self.account.get_holdings(), {})
        self.assertEqual(self.account.get_portfolio_value(), Decimal('0'))
        self.assertEqual(self.account.get_lifetime_pnl(), Decimal('0'))
        self.assertEqual(self.account.get_transactions(), [])
    
    def test_deposit_positive(self):
        tx = self.account.deposit(Decimal('1000'))
        self.assertEqual(self.account.get_cash_balance(), Decimal('1000'))
        self.assertEqual(tx.cash_delta, Decimal('1000'))
        self.assertEqual(tx.note, 'Deposit $1000')
        self.assertEqual(self.account._initial_deposit, Decimal('1000'))
    
    def test_deposit_negative_amount(self):
        with self.assertRaises(ValueError):
            self.account.deposit(Decimal('-100'))
    
    def test_deposit_zero(self):
        with self.assertRaises(ValueError):
            self.account.deposit(Decimal('0'))
    
    def test_deposit_non_decimal(self):
        tx = self.account.deposit(1000)
        self.assertEqual(self.account.get_cash_balance(), Decimal('1000'))
        
        tx = self.account.deposit(500.50)
        self.assertEqual(self.account.get_cash_balance(), Decimal('1500.50'))
    
    def test_withdraw_sufficient_funds(self):
        self.account.deposit(Decimal('1000'))
        tx = self.account.withdraw(Decimal('500'))
        self.assertEqual(self.account.get_cash_balance(), Decimal('500'))
        self.assertEqual(tx.cash_delta, Decimal('-500'))
        self.assertEqual(tx.note, 'Withdraw $500')
    
    def test_withdraw_insufficient_funds(self):
        self.account.deposit(Decimal('100'))
        with self.assertRaises(ValueError):
            self.account.withdraw(Decimal('200'))
    
    def test_withdraw_negative_amount(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(Decimal('-100'))
    
    def test_buy_sufficient_funds(self):
        self.account.deposit(Decimal('2000'))
        tx = self.account.buy('AAPL', Decimal('10'))
        self.assertEqual(self.account.get_cash_balance(), Decimal('500'))  # 2000 - 10*150 = 500
        self.assertEqual(self.account.get_holdings(), {'AAPL': Decimal('10')})
        self.assertEqual(tx.symbol, 'AAPL')
        self.assertEqual(tx.quantity, Decimal('10'))
        self.assertEqual(tx.price, Decimal('150'))
        self.assertEqual(tx.cash_delta, Decimal('-1500'))
        self.assertEqual(tx.note, 'Buy 10 shares of AAPL at $150 per share')
    
    def test_buy_insufficient_funds(self):
        self.account.deposit(Decimal('100'))
        with self.assertRaises(ValueError):
            self.account.buy('AAPL', Decimal('10'))
    
    def test_buy_negative_quantity(self):
        with self.assertRaises(ValueError):
            self.account.buy('AAPL', Decimal('-10'))
    
    def test_sell_sufficient_holdings(self):
        self.account.deposit(Decimal('2000'))
        self.account.buy('AAPL', Decimal('10'))
        tx = self.account.sell('AAPL', Decimal('5'))
        self.assertEqual(self.account.get_cash_balance(), Decimal('1250'))  # 500 + 5*150 = 1250
        self.assertEqual(self.account.get_holdings(), {'AAPL': Decimal('5')})
        self.assertEqual(tx.symbol, 'AAPL')
        self.assertEqual(tx.quantity, Decimal('-5'))
        self.assertEqual(tx.price, Decimal('150'))
        self.assertEqual(tx.cash_delta, Decimal('750'))
        self.assertEqual(tx.note, 'Sell 5 shares of AAPL at $150 per share')
    
    def test_sell_insufficient_holdings(self):
        self.account.deposit(Decimal('2000'))
        self.account.buy('AAPL', Decimal('5'))
        with self.assertRaises(ValueError):
            self.account.sell('AAPL', Decimal('10'))
    
    def test_sell_all_holdings(self):
        self.account.deposit(Decimal('2000'))
        self.account.buy('AAPL', Decimal('5'))
        self.account.sell('AAPL', Decimal('5'))
        self.assertEqual(self.account.get_holdings(), {})
    
    def test_portfolio_value(self):
        self.account.deposit(Decimal('1000'))
        self.account.buy('AAPL', Decimal('2'))
        # Cash: 1000 - 2*150 = 700
        # Holdings: 2 * 150 = 300
        # Total: 700 + 300 = 1000
        self.assertEqual(self.account.get_portfolio_value(), Decimal('1000'))
        
        # Test with multiple symbols
        self.account.buy('TSLA', Decimal('1'))
        # Cash: 700 - 800 = -100
        # AAPL: 2 * 150 = 300
        # TSLA: 1 * 800 = 800
        # Total: -100 + 300 + 800 = 1000
        self.assertEqual(self.account.get_portfolio_value(), Decimal('1000'))
    
    def test_lifetime_pnl(self):
        self.account.deposit(Decimal('1000'))
        self.assertEqual(self.account.get_lifetime_pnl(), Decimal('0'))
        
        self.account.buy('AAPL', Decimal('2'))
        # Portfolio value: 700 + 300 = 1000, same as initial deposit
        self.assertEqual(self.account.get_lifetime_pnl(), Decimal('0'))
        
        # Sell at same price - no P&L
        self.account.sell('AAPL', Decimal('1'))
        # Cash: 700 + 150 = 850
        # Holdings: 1 * 150 = 150
        # Portfolio value: 1000
        self.assertEqual(self.account.get_lifetime_pnl(), Decimal('0'))
    
    def test_get_transactions(self):
        self.assertEqual(self.account.get_transactions(), [])
        
        tx1 = self.account.deposit(Decimal('1000'))
        tx2 = self.account.buy('AAPL', Decimal('2'))
        
        transactions = self.account.get_transactions()
        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0], tx1)
        self.assertEqual(transactions[1], tx2)
        
        # Test limit
        limited = self.account.get_transactions(1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0], tx2)
    
    def test_symbol_pnl(self):
        self.account.deposit(Decimal('2000'))
        self.account.buy('AAPL', Decimal('10'))  # Cash delta: -1500
        self.account.sell('AAPL', Decimal('5'))  # Cash delta: +750
        
        # P&L for AAPL: -1500 + 750 = -750
        self.assertEqual(self.account.get_symbol_pnl('AAPL'), Decimal('-750'))
        
        # Test unknown symbol
        self.assertEqual(self.account.get_symbol_pnl('UNKNOWN'), Decimal('0'))
        
        # Test with multiple symbols
        self.account.buy('TSLA', Decimal('1'))  # Cash delta: -800
        self.account.sell('TSLA', Decimal('1'))  # Cash delta: +800
        
        self.assertEqual(self.account.get_symbol_pnl('TSLA'), Decimal('0'))

    def test_initial_deposit_tracking(self):
        # First deposit should set initial_deposit
        self.account.deposit(Decimal('500'))
        self.assertEqual(self.account._initial_deposit, Decimal('500'))
        
        # Subsequent deposits shouldn't change initial_deposit
        self.account.deposit(Decimal('300'))
        self.assertEqual(self.account._initial_deposit, Decimal('500'))

if __name__ == '__main__':
    unittest.main()