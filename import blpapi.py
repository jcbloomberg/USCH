import blpapi
from blpapi import SessionOptions, Session

# Step 1: Set up session options
options = SessionOptions()
options.setServerHost('localhost')
options.setServerPort(8194)

# Step 2: Start session
session = Session(options)
if not session.start():
    print("❌ Failed to start session.")
    exit()
print("✅ Session started.")

# Step 3: Open the refdata service
if not session.openService("//blp/refdata"):
    print("❌ Failed to open //blp/refdata service.")
    exit()
print("✅ Service opened.")

# Step 4: Create the request
service = session.getService("//blp/refdata")
request = service.createRequest("ReferenceDataRequest")
request.getElement("securities").appendValue("AAPL US Equity")
request.getElement("fields").appendValue("PX_LAST")  # Last price

# Step 5: Send the request
session.sendRequest(request)

# Step 6: Wait for response
while True:
    ev = session.nextEvent()
    for msg in ev:
        print(msg)
    if ev.eventType() == blpapi.Event.RESPONSE:
        break
