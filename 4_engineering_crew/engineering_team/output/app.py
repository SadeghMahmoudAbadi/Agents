import gradio as gr
from accounts import Account, get_share_price
from decimal import Decimal

# Initialize a single account for the demo
account = Account("Demo User")

def format_decimal(d):
    return f"${float(d):.2f}"

def create_account():
    return "Account created successfully!"

def deposit(amount):
    try:
        amount = float(amount)
        tx = account.deposit(amount)
        return f"Deposited {format_decimal(tx.cash_delta)}. New balance: {format_decimal(account.get_cash_balance())}"
    except Exception as e:
        return f"Error: {str(e)}"

def withdraw(amount):
    try:
        amount = float(amount)
        tx = account.withdraw(amount)
        return f"Withdrew {format_decimal(-tx.cash_delta)}. New balance: {format_decimal(account.get_cash_balance())}"
    except Exception as e:
        return f"Error: {str(e)}"

def buy_shares(symbol, quantity):
    try:
        quantity = float(quantity)
        tx = account.buy(symbol, quantity)
        return f"Bought {quantity} shares of {symbol} for {format_decimal(-tx.cash_delta)}. Cash balance: {format_decimal(account.get_cash_balance())}"
    except Exception as e:
        return f"Error: {str(e)}"

def sell_shares(symbol, quantity):
    try:
        quantity = float(quantity)
        tx = account.sell(symbol, quantity)
        return f"Sold {quantity} shares of {symbol} for {format_decimal(tx.cash_delta)}. Cash balance: {format_decimal(account.get_cash_balance())}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_balance():
    return f"Cash Balance: {format_decimal(account.get_cash_balance())}"

def get_holdings():
    holdings = account.get_holdings()
    if not holdings:
        return "No holdings"
    
    result = "Holdings:\n"
    for symbol, quantity in holdings.items():
        result += f"{symbol}: {float(quantity):.2f} shares\n"
    return result

def get_portfolio_value():
    value = account.get_portfolio_value()
    return f"Portfolio Value: {format_decimal(value)}"

def get_pnl():
    pnl = account.get_lifetime_pnl()
    return f"Profit/Loss: {format_decimal(pnl)}"

def get_transactions():
    transactions = account.get_transactions()
    if not transactions:
        return "No transactions"
    
    result = "Transactions (newest last):\n"
    for tx in transactions:
        result += f"{tx.note}\n"
    return result

def get_symbol_pnl(symbol):
    try:
        pnl = account.get_symbol_pnl(symbol)
        return f"P&L for {symbol}: {format_decimal(pnl)}"
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Trading Account Demo") as demo:
    gr.Markdown("# Trading Account Management System")
    gr.Markdown("A simple demo of the trading account backend")
    
    with gr.Tab("Account Operations"):
        with gr.Row():
            with gr.Column():
                deposit_amount = gr.Number(label="Deposit Amount", value=1000)
                deposit_btn = gr.Button("Deposit")
                deposit_output = gr.Textbox(label="Deposit Result")
                
                withdraw_amount = gr.Number(label="Withdraw Amount", value=100)
                withdraw_btn = gr.Button("Withdraw")
                withdraw_output = gr.Button("Withdraw")
                withdraw_output = gr.Textbox(label="Withdraw Result")
                
            with gr.Column():
                balance_output = gr.Textbox(label="Cash Balance")
                balance_btn = gr.Button("Get Balance")
                
                portfolio_value_output = gr.Textbox(label="Portfolio Value")
                portfolio_value_btn = gr.Button("Get Portfolio Value")
                
                pnl_output = gr.Textbox(label="Profit/Loss")
                pnl_btn = gr.Button("Get P&L")
    
    with gr.Tab("Trading"):
        with gr.Row():
            with gr.Column():
                buy_symbol = gr.Dropdown(
                    choices=["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"],
                    label="Symbol to Buy",
                    value="AAPL"
                )
                buy_quantity = gr.Number(label="Quantity to Buy", value=1)
                buy_btn = gr.Button("Buy Shares")
                buy_output = gr.Textbox(label="Buy Result")
                
            with gr.Column():
                sell_symbol = gr.Dropdown(
                    choices=["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"],
                    label="Symbol to Sell",
                    value="AAPL"
                )
                sell_quantity = gr.Number(label="Quantity to Sell", value=1)
                sell_btn = gr.Button("Sell Shares")
                sell_output = gr.Textbox(label="Sell Result")
    
    with gr.Tab("Reports"):
        with gr.Row():
            with gr.Column():
                holdings_output = gr.Textbox(label="Current Holdings")
                holdings_btn = gr.Button("Get Holdings")
                
                transactions_output = gr.Textbox(label="Transaction History")
                transactions_btn = gr.Button("Get Transactions")
                
            with gr.Column():
                symbol_pnl_input = gr.Dropdown(
                    choices=["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"],
                    label="Symbol for P&L",
                    value="AAPL"
                )
                symbol_pnl_btn = gr.Button("Get Symbol P&L")
                symbol_pnl_output = gr.Textbox(label="Symbol P&L Result")
    
    # Event handlers
    deposit_btn.click(
        fn=deposit,
        inputs=deposit_amount,
        outputs=deposit_output
    )
    
    withdraw_btn.click(
        fn=withdraw,
        inputs=withdraw_amount,
        outputs=withdraw_output
    )
    
    balance_btn.click(
        fn=get_balance,
        inputs=[],
        outputs=balance_output
    )
    
    portfolio_value_btn.click(
        fn=get_portfolio_value,
        inputs=[],
        outputs=portfolio_value_output
    )
    
    pnl_btn.click(
        fn=get_pnl,
        inputs=[],
        outputs=pnl_output
    )
    
    buy_btn.click(
        fn=buy_shares,
        inputs=[buy_symbol, buy_quantity],
        outputs=buy_output
    )
    
    sell_btn.click(
        fn=sell_shares,
        inputs=[sell_symbol, sell_quantity],
        outputs=sell_output
    )
    
    holdings_btn.click(
        fn=get_holdings,
        inputs=[],
        outputs=holdings_output
    )
    
    transactions_btn.click(
        fn=get_transactions,
        inputs=[],
        outputs=transactions_output
    )
    
    symbol_pnl_btn.click(
        fn=get_symbol_pnl,
        inputs=symbol_pnl_input,
        outputs=symbol_pnl_output
    )

if __name__ == "__main__":
    demo.launch()