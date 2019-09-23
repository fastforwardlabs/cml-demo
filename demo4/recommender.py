import joblib

local_df = joblib.load('clickstream.pkl')
all_products = local_df['product'].unique()
known_ratings = local_df.groupby(['ip', 'product']).size()

algo = joblib.load('recommender.pkl')

def recommend(data):
  """
  For a given IP address, use surprise to recommend the
  product with the highest predicted rating.

  Input shape: {"ip": "99.99.191.106"}
  Output shape: {"recommendation": "The North Face Women's Recon Backpack"}
  """
  ip_to_recommend = data['ip']
  max_rating_est = 0
  recommendation = "unknown"
  for product in all_products:
    if known_ratings.get((ip_to_recommend, product)) is None:
      rating_est = algo.predict(ip_to_recommend, product).est
      if rating_est > max_rating_est:
        recommendation = product
        max_rating_est = rating_est
  return {"recommendation": recommendation}

#recommend({"ip": "99.99.191.106"})