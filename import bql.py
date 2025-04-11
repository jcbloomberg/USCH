import blpapi
from blpapi import SessionOptions, Session # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from arch import arch_model

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

# -------- Main: Tickers & Modeling --------
tickers = {
    "SCHW": "SCHW US Equity",
    "JPM": "JPM US Equity",
    "BAC": "BAC US Equity",
    "C": "C US Equity",
    "WFC": "WFC US Equity",
    "GS": "GS US Equity",
    "MS": "MS US Equity",
    "USB": "USB US Equity",
    "CCB": "939 HK Equity",
    "BOC": "3988 HK Equity",
    "BOCOM": "3328 HK Equity",
    "CMB": "3968 HK Equity",
}

start = datetime.date(2004, 1, 1)
end = datetime.date.today()

plt.figure(figsize=(14, 8))
for name, ticker in tickers.items():
    try:
        df = fetch_bloomberg_data(ticker, start, end)
        df["log_return"] = np.log(df["PX_LAST"] / df["PX_LAST"].shift(1)) * 100
        df.dropna(inplace=True)
        model = arch_model(df["log_return"], vol='GARCH', p=1, q=1)
        res = model.fit(disp="off")
        df["garch_vol"] = res.conditional_volatility
        plt.plot(df.index, df["garch_vol"], label=name)
    except Exception as e:
        print(f"⚠️ Failed for {name}: {e}")

plt.title("GARCH(1,1) Estimated Volatility of Global Banks")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
