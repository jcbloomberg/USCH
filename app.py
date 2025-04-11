import blpapi
from blpapi import SessionOptions, Session
import pandas as pd
import numpy as np
from arch import arch_model
import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# -------- Bloomberg Pull --------
def fetch_bloomberg_data(ticker, start_date, end_date):
    options = SessionOptions()
    options.setServerHost("localhost")
    options.setServerPort(8194)
    session = Session(options)

    if not session.start():
        raise RuntimeError("Could not start Bloomberg session.")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Could not open //blp/refdata")

    service = session.getService("//blp/refdata")
    request = service.createRequest("HistoricalDataRequest")
    request.getElement("securities").appendValue(ticker)
    request.getElement("fields").appendValue("PX_LAST")
    request.set("startDate", start_date.strftime("%Y%m%d"))
    request.set("endDate", end_date.strftime("%Y%m%d"))
    request.set("periodicitySelection", "DAILY")
    session.sendRequest(request)

    data = []
    while True:
        ev = session.nextEvent()
        for msg in ev:
            if msg.hasElement("securityData"):
                sec_data = msg.getElement("securityData")
                field_data = sec_data.getElement("fieldData")
                for i in range(field_data.numValues()):
                    row = field_data.getValue(i)
                    data.append({
                        "date": row.getElementAsDatetime("date"),
                        "PX_LAST": row.getElementAsFloat("PX_LAST")
                    })
        if ev.eventType() == blpapi.Event.RESPONSE:
            break

    df = pd.DataFrame(data).set_index("date")
    return df

# -------- Tickers --------
tickers = {
    "Charles Schwab (SCHW)": "SCHW US Equity",
    "JPMorgan Chase (JPM)": "JPM US Equity",
    "Bank of America (BAC)": "BAC US Equity",
    "Citigroup (C)": "C US Equity",
    "Wells Fargo (WFC)": "WFC US Equity",
    "Goldman Sachs (GS)": "GS US Equity",
    "Morgan Stanley (MS)": "MS US Equity",
    "U.S. Bancorp (USB)": "USB US Equity",
    "China Construction Bank (CCB)": "939 HK Equity",
    "Bank of China (BOC)": "3988 HK Equity",
    "Bank of Communications (BoCom)": "3328 HK Equity",
    "China Merchants Bank (CMB)": "3968 HK Equity"
}

# -------- Data Preprocessing & Caching --------
start = datetime.date(2022, 1, 1)
end = datetime.date.today()
vol_data = {}

for name, ticker in tickers.items():
    try:
        df = fetch_bloomberg_data(ticker, start, end)
        df["log_return"] = np.log(df["PX_LAST"] / df["PX_LAST"].shift(1)) * 100
        df.dropna(inplace=True)
        model = arch_model(df["log_return"], vol='GARCH', p=1, q=1)
        res = model.fit(disp="off")
        df["garch_vol"] = res.conditional_volatility
        vol_data[name] = df["garch_vol"]
    except Exception as e:
        print(f"⚠️ {name}: {e}")

# -------- Dash App --------
app = dash.Dash(__name__)
app.title = "Global Bank Volatility (GARCH)"

app.layout = html.Div([
    html.H1("📉 Global Bank Volatility Dashboard (GARCH)", style={"textAlign": "center"}),
    dcc.Dropdown(
        id="bank-selector",
        options=[{"label": name, "value": name} for name in vol_data],
        value=["JPMorgan Chase (JPM)"],
        multi=True
    ),
    dcc.Graph(id="vol-graph", style={"height": "700px"})
])

@app.callback(
    Output("vol-graph", "figure"),
    [Input("bank-selector", "value")]
)
def update_graph(selected_banks):
    traces = []
    for bank in selected_banks:
        trace = go.Scatter(
            x=vol_data[bank].index,
            y=vol_data[bank],
            mode="lines",
            name=bank
        )
        traces.append(trace)

    return {
        "data": traces,
        "layout": go.Layout(
            title="GARCH(1,1) Volatility Over Time",
            xaxis={"title": "Date"},
            yaxis={"title": "Volatility (%)"},
            hovermode="closest"
        )
    }

if __name__ == "__main__":
    app.run(debug=True)
