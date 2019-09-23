import joblib
import pandas as pd

from sklearn.cluster import KMeans

df = joblib.load("clickstream.pkl")

user_items = (pd.get_dummies(df["product"])
  .groupby(df["ip"]).apply(max))

model = KMeans(n_clusters=5)

model.fit(user_items)

label = pd.DataFrame(
  model.predict(user_items),
  index=user_items.index
)