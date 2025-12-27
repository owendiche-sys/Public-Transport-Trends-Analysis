import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FuncFormatter

# Read the data
df2019 = pd.read_csv("2019data0.csv")
df2022 = pd.read_csv("2022data0.csv")

#FIGURE 1 CODE
# 2019 Daily Total Passenger Computation
df2019["Date"] = pd.to_datetime(df2019["Date"])

bus_cols  = ["Bus pax number peak","Bus pax number offpeak"]
tram_cols = ["Tram pax number peak","Tram pax number offpeak"]
metro_cols= ["Metro pax number peak","Metro pax number offpeak"]

df2019["Total passengers"] =(
    df2019[bus_cols + tram_cols + metro_cols].sum(axis=1)
)    

df2019 = df2019.sort_values("Date")

y2019 = df2019["Total passengers"].values
x2019 = np.arange(1 , len(y2019)+1)

# Expansion of 2022 Sample to Annual Population
# Convert timestamp
df2022["Date and time"] = pd.to_datetime(df2022["Date and time"])
df2022["Date"] = df2022["Date and time"].dt.date

# daily counts = sample counts
daily2022 = df2022.groupby("Date").size().reset_index(name="Sample")

# Total annual passengers from assignment table
total_2022 = 369382078

sample_total = daily2022["Sample"].sum()

# expansion factor
expansion_factor = total_2022 / sample_total

# scale sample to estimate real totals
daily2022["Total passengers"] = daily2022["Sample"] * expansion_factor

daily2022 = daily2022.sort_values("Date")

y2022 = daily2022["Total passengers"].values
x2022 = np.arange(1, len(y2022)+1)


# Fourier smoothing (8 terms)
def fourier_smooth(y, n_terms=8):
    N = len(y)
    fft_vals = np.fft.rfft(y)
    fft_vals[n_terms+1:] = 0
    return np.fft.irfft(fft_vals, n=N)

y2019_smooth = fourier_smooth(y2019,8)
y2022_smooth = fourier_smooth(y2022,8)


# Figure 1 plot
plt.figure(figsize=(12,6))
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='y')
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

# scatter
ax.scatter(
    x2019, 
    y2019, 
    s=40, 
    alpha=0.7, 
    label="2019",
    color="Red",
    marker='o'
)

ax.scatter(
    x2022, 
    y2022, 
    s=40, 
    alpha=0.7, 
    label="2022",
    color="black",
    marker='x'
)

# smooth lines
ax.plot(
        x2019, 
        y2019_smooth, 
        linewidth=3, 
        label="2019 Fourier",
        color="Red"
)

ax.plot(
        x2022,
        y2022_smooth, 
        linewidth=3,
        label="2022 Fourier",
        color="Black"
)

ax.set_xlabel(
    "Day of the year", 
    fontsize=16, 
    fontweight="bold"
)

ax.set_ylabel(
    "Total Daily Passengers", 
    fontsize=16, 
    fontweight="bold"
)

ax.set_title(
    "Daily Public Transport Passengers (2019 & 2022)",
    fontsize=16,
    fontweight="bold"
)

ax.legend(loc="upper center",
          #bbox_to_anchor=(1, 0.6),
          fontsize=16,
          ncol=4,
          frameon=True,
          edgecolor="black")

ax.grid(True, linestyle="--", alpha=0.3)

# student ID
ax.text(0.79,0.02,  
        "Student ID:24152506",
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        bbox=dict(facecolor="white", edgecolor="black"))

plt.xticks(
    fontsize=14,
    fontweight='bold'
)

plt.yticks(
    fontsize=14,
    fontweight='bold'
)

plt.tight_layout()
plt.savefig("Figure 1.png",dpi=300)
plt.show()


#FIGURE 2 CODE
# Ensure Date is datetime for both datasets
df2019["Date"] = pd.to_datetime(df2019["Date"])
df2019["Day"] = df2019["Date"].dt.day_name()

# Convert day names correctly:
daily2022["Day"] = pd.to_datetime(daily2022["Date"]).dt.day_name()

# Order of weekdays
order = ["Monday",
         "Tuesday",
         "Wednesday",
         "Thursday",
         "Friday",
         "Saturday",
         "Sunday"]

# Average Daily Passengers
avg2019 = df2019.groupby("Day")["Total passengers"].mean().reindex(order)
avg2022 = daily2022.groupby("Day")["Total passengers"].mean().reindex(order)


# X, Y, Z percentages
df2022["Month"] = pd.to_datetime(df2022["Date"]).dt.month

spring = df2022[df2022["Month"].isin([3,4,5])].shape[0]
summer = df2022[df2022["Month"].isin([6,7,8])].shape[0]
autumn = df2022[df2022["Month"].isin([9,10,11])].shape[0]
total2022 = df2022.shape[0]

X = spring / total2022 * 100
Y = summer / total2022 * 100
Z = autumn / total2022 * 100


# Figure 2 Plot
plt.figure(figsize=(12,6))
x = np.arange(len(order))
width = 0.35

plt.bar(
        x - width/2, 
        avg2019,
        width, 
        label="2019",
        color="Red",       
        edgecolor="black"
)
plt.bar(
        x + width/2, 
        avg2022,
        width, 
        label="2022",
        color="Black",
        edgecolor="black"
)

plt.xticks(x, order, fontsize=14, fontweight="bold")
plt.ylabel("Average Daily Passengers", fontsize=16, fontweight="bold")
plt.xlabel("Day of Week", fontsize=16, fontweight="bold")
plt.title(
    "Average Daily Passengers by Day of Week (2019 & 2022)", 
    fontsize=18,
    fontweight="bold"
)

ax = plt.gca()
ax.ticklabel_format(style='plain', axis='y')
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.legend(
    fontsize=14,
    frameon=True, 
    edgecolor="black"
)
plt.yticks(
    fontweight='bold',
    fontsize =14
)

# Text box
text_box = (
    f"Student ID: 24152506\n"
    f"Spring (X): {X:.2g}%\n"
    f"Summer (Y): {Y:.2g}%\n"
    f"Autumn (Z): {Z:.2g}%"
)

plt.text(
    0.73, 0.80, text_box,
    transform=plt.gca().transAxes,
    fontsize=15,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.85)
)

plt.tight_layout()
plt.savefig("Figure 2.png", dpi=300)
plt.show()

#FIGURE 3 CODE
# Filtering for metro journeys only
metro2022 = df2022[df2022["Mode"] == "Metro"].copy()

# Extracting variables
X = metro2022["Distance"].values.reshape(-1, 1)
y = metro2022["Price"].values

# Linear Regression Fit
model = LinearRegression()
model.fit(X, y)

a = model.coef_[0]      # slope
b = model.intercept_    # intercept

# Regression line for plotting
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)


# Figure 3 Plot
plt.figure(figsize=(12, 6))

plt.scatter(
    X, 
    y, 
    s=50, 
    alpha=0.7, 
    label="Metro journeys",
    color='Black',
    marker='x'
)
plt.plot(
    x_line, 
    y_line, 
    color="red", 
    linewidth=3, 
    label="Linear fit"
)
plt.xlabel(
    "Trip length (km)",
    fontsize=16, 
    fontweight="bold"
)
plt.ylabel(
    "Price (€)", 
    fontsize=16, 
    fontweight="bold"
)
plt.title(
    "Price vs Trip Length for 2022 Metro Journeys", 
    fontsize=18,
    fontweight="bold"
)
plt.xticks(fontsize=14, fontweight="bold")
plt.yticks(fontsize=14, fontweight="bold")

plt.legend(
    fontsize=18,
    frameon=True,
    edgecolor="black",
    loc="upper left"
)

# Display formula on the plot
formula_text = f"Fit: Price = {a:.3f} × Distance + {b:.3f}"
plt.text(
    0.5, 0.15,
    f"Fit: Price = {a:.3f} × Distance + {b:.3f}\nStudent ID: 24152506",
    transform=plt.gca().transAxes,
    fontsize=16,
    ha="center",
    va="center",
    fontweight="bold",
    bbox=dict(
        facecolor="white", 
        edgecolor="black",
        linewidth=1.5, 
        alpha=0.9
        )
)
plt.tight_layout()
plt.savefig("Figure 3.png", dpi=300)
plt.show()

#FIGURE 4 CODE
# Fraction of journey by mode (2019)
# Sum all 2019 passenger counts
bus2019 = df2019[["Bus pax number peak", 
                  "Bus pax number offpeak"]].sum().sum()
tram2019 = df2019[["Tram pax number peak", 
                   "Tram pax number offpeak"]].sum().sum()
metro2019 = df2019[["Metro pax number peak", 
                    "Metro pax number offpeak"]].sum().sum()

total2019 = bus2019 + tram2019 + metro2019

fractions2019 = {
    "Bus": bus2019 / total2019 * 100,
    "Tram": tram2019 / total2019 * 100,
    "Metro": metro2019 / total2019 * 100
}

# Fraction of journey by mode (2022)
# Count each mode as % of total trips
fractions2022 = df2022["Mode"].value_counts(normalize=True) * 100

# Ensureing all three modes exist
vals2022 = [
    fractions2022.get("Bus", 0),
    fractions2022.get("Tram", 0),
    fractions2022.get("Metro", 0)
]

# 2019 values in correct order
vals2019 = [
    fractions2019["Bus"],
    fractions2019["Tram"],
    fractions2019["Metro"]
]

# Calculate Seasonal Fractions (X, Y, Z)
df2022["Date and time"] = pd.to_datetime(df2022["Date and time"])
df2022["Month"] = df2022["Date and time"].dt.month

spring = df2022[df2022["Month"].isin([3, 4, 5])].shape[0]
summer = df2022[df2022["Month"].isin([6, 7, 8])].shape[0]
autumn = df2022[df2022["Month"].isin([9, 10, 11])].shape[0]
total2022 = df2022.shape[0]

X = spring / total2022 * 100
Y = summer / total2022 * 100
Z = autumn / total2022 * 100

print("Spring %:", X)
print("Summer %:", Y)
print("Autumn %:", Z)

# FIGURE 4 PLOT

labels = ["Bus", "Tram", "Metro"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width/2, 
    vals2019, 
    width, 
    label="2019",
    edgecolor="black", 
    linewidth=1.3,
    color='Red'
)
plt.bar(
    x + width/2,
    vals2022, 
    width, 
    label="2022",
    edgecolor="black", 
    linewidth=1.3,
    color='Black'
)
plt.xlabel(
    "Mode of Transport",
    fontsize=16, 
    fontweight="bold"
)
plt.ylabel(
    "Percentage of journeys (%)",
    fontsize=16, 
    fontweight="bold"
)
plt.title(
    "Fraction of journeys by transport mode (2019 & 2022)",
    fontsize=18, 
    fontweight="bold",
)
plt.xticks(
    x, labels, 
    fontsize=14, 
    fontweight="bold"
)
plt.yticks(
    fontsize=14,
    fontweight="bold"
)
plt.legend(
    fontsize=16,
    frameon=True,
    edgecolor="black"
)

# Student ID
plt.text(
    0.05, 0.95,
    "Student ID: 24152506",
    transform=plt.gca().transAxes,
    fontsize=16,
    fontweight="bold",
    bbox=dict(facecolor="white", edgecolor="black", alpha=0.8)
)

plt.tight_layout()
plt.savefig("Figure 4.png", dpi=300)
plt.show()