import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from plotly import graph_objects as go
from datetime import date
import seaborn as sns

start = "1993-01-01"

today = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")


stocks = sorted(("^GSPC", "AMZN", 'BRK-B',"NVDA",
                  "MSFT",  "TSM", "META", "WMT", "V"))

selected_stock = st.selectbox("Select dataset for prediction", stocks)



@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, today)

    # data = yf.download(ticker, start, today, auto_adjust=True)
    data.reset_index(inplace=True)
    # drop the "Stock name" level so your columns become simple again
    data.columns = data.columns.get_level_values(0)
    # Remove the column by name
    data.drop(columns={"Dividends", "Stock Splits", "Adj Close"},  inplace=True, errors='ignore')
    # remove timezone (00:00:00)
    data['Date'] = pd.to_datetime(data['Date']).dt.date

    data = data.loc["1990-01-01":].copy()


    return data



load_data_state = st.text("Load data....")
data = load_data(selected_stock)
load_data_state.text("Loading data.... done!")



def Main_figure():

    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    from datetime import timedelta

    # --- MOCK DATA SETUP ---
    # (Assume 'df' is your dataframe with 'Date', 'stock_open', and 'stock_close')
    # df = pd.read_csv('your_data.csv') 

    # Initialize zoom level in session state
    if 'zoom_factor' not in st.session_state:
        st.session_state.zoom_factor = 1.0

    # Create the layout with a small column for buttons and a large one for the chart
    col1, col2 = st.columns([0.05, 0.95])

    with col1:
        st.write("##") # Offset for alignment
        if st.button("➕", help="Zoom In"):
            st.session_state.zoom_factor *= 0.8
        if st.button("➖", help="Zoom Out"):
            st.session_state.zoom_factor *= 1.2
        if st.button("↺", help="Reset"):
            st.session_state.zoom_factor = 1.0
    # Calculate the new Date Range based on zoom factor
    end_date = data['Date'].max()
    start_date = data['Date'].min()
    total_duration = end_date - start_date
    actual_duration = total_duration * st.session_state.zoom_factor
    new_start = end_date - actual_duration

    # --- PLOTLY CHART ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))

    # Apply the zoom to the X-axis
    fig.update_layout(
        xaxis_range=[new_start, end_date],
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    col2.plotly_chart(fig, use_container_width=True)
Main_figure()


st.write("-----------------------------------------------------")

st.subheader(f"Raw data")
st.write(data.tail())

# -----------------------------------------------------------------------------------------------------------------------
# data preprocessing


data.set_index("Date", inplace=True)

data["Tomorrow"] = data["Close"].shift(-1)

data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

# 1. احتفظ بأحدث سعر متاح فعلياً قبل حذف الـ NaN
real_latest_price = data['Close'].iloc[-1]
real_latest_date = data.index[-1]


# ------------------------------------
# 1. Download VIX
vix_raw = yf.download("^VIX", start=data.index[0])

# 2. Extract only the 'Close' column and flatten it
# This ensures we have a simple column named "VIX"
if "Close" in vix_raw.columns:
    data["VIX"] = vix_raw["Close"]
else:
    # Handles cases where yfinance returns multi-index columns
    data["VIX"] = vix_raw.iloc[:, 0] 

# 3. Fill missing values (weekends/holidays)
data["VIX"] = data["VIX"].ffill()

# --- NOW START YOUR LOOP ---
horizens = [2, 5, 60]
new_predictores = []

for horizon in horizens:
    rolling_avgs = data.rolling(horizon).mean()
    
    # S&P 500 Ratios
    ratio_column = f"Close_Ratio_{horizon}"
    data[ratio_column] = data["Close"] / rolling_avgs["Close"]
    
    # VIX Ratios - THIS WILL NOW WORK
    vix_ratio_column = f"VIX_Ratio_{horizon}"
    data[vix_ratio_column] = data["VIX"] / rolling_avgs["VIX"]
    
    # Trend
    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictores += [ratio_column, trend_column, vix_ratio_column]

data.dropna(inplace=True)


# build the model--------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix



model = RandomForestClassifier(
    n_estimators=200, 
    min_samples_split=100, 
    random_state=1,
    n_jobs=-1 # لاستخدام كل معالجات الجهاز لتسريع العملية
)


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    
    # الحصول على الاحتمالات بدلاً من 0 أو 1 مباشرة
    # هذا يسمح لنا بالتحكم في "قوة" الإشارة
    preds_prob = model.predict_proba(test[predictors])[:, 1]
    
    # لن نعتبره "صعود" إلا إذا كانت الثقة أكبر من 55%
    preds = (preds_prob >= .55).astype(int)
    
    combined = pd.concat([test["Target"], pd.Series(preds, index=test.index, name="Predictions")], axis=1)
    return combined
st.write("----------------------------------------------------------")

def backtest(data, new_predictores, model, start=1250, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):

        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        predictions = predict(train, test, new_predictores, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)



preds = backtest(data, new_predictores, model)

def tomorrow_preds(latest_price, latest_date):
    col1, col2 = st.columns(2)
    with col1:
        # استخدام السعر الذي حفظناه قبل الـ dropna
        st.metric("Latest Close Price", f"${latest_price:.2f}")
        
    with col2:
        # التوقع يبقى كما هو من مصفوفة preds
        last_pred = "UP 📈" if preds["Predictions"].iloc[-1] == 1 else "DOWN 📉"
        st.metric("Model Prediction for Tomorrow", last_pred)
        st.caption(f"Based on data up to: {latest_date}")

# استدعاء الدالة بالمتغيرات الجديدة
tomorrow_preds(real_latest_price, real_latest_date)

st.write("---------------------------------------------------------")

st.write("# rows in the back tested data", preds.shape[0])

st.write("---------------------------------------------------------")


def moes():

    st.write("# Evaluate the model: ")
    st.write("## precision: ", round(precision_score(preds["Target"], preds["Predictions"]), 3))
    st.write("## accuracy: ", round(accuracy_score(preds["Target"], preds["Predictions"]), 3))
    st.write("## recall: ", round(recall_score(preds["Target"], preds["Predictions"]), 3))
    st.write("## f1 score: ", round(f1_score(preds["Target"], preds["Predictions"]), 3))



    st.write("----------------------------------------------------------")
moes()

# Assuming 'y_test' and 'y_pred' are your variables
cm = confusion_matrix(preds["Target"], preds["Predictions"])

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')

# Use st.pyplot to display it in Streamlit
st.pyplot(fig)

st.write("----------------------------------------------------------")



def percision_fig():
    
    import plotly.graph_objects as go

    # 1. Filter data: only look at cases where the model predicted 'Up' (1)
    buy_signals = preds[preds["Predictions"] == 1]

    # 2. Count the actual outcomes for those specific signals
    counts = buy_signals["Target"].value_counts().sort_index() 
    # counts[0] = False Positives (Predicted Up, actually Down)
    # counts[1] = True Positives (Predicted Up, actually Up)

    # 3. Create the Bar Chart
    fig_bar = go.Figure(data=[
        go.Bar(
            x=['Wrong (Price went Down)', 'Right (Price went Up)'],
            y=[counts.get(0, 0), counts.get(1, 0)],
            marker_color=['#ef553b', '#00cc96'] # Red and Green
        )
    ])

    fig_bar.update_layout(
        title="Outcome of 'Price Up' Predictions",
        yaxis_title="Number of Days",
        template="plotly_dark"
    )

    st.plotly_chart(fig_bar)
percision_fig()



importance = pd.Series(model.feature_importances_, index=new_predictores).sort_values(ascending=False)


st.write("# Feature Importance:", importance)



