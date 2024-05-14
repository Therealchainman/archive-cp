# Pandas Tricks

## Best syntax to perform groupby aggregation

here is an example,  you can provide the agg as tuples with names. 

why is sort = False
reset_index will of course, reset the index to the aggregated data output

```py
def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:
  df = (
    activities
    .sort_values("sell_date")
    .groupby("sell_date", sort = False)
    .agg(
      num_sold = ("product", "nunique"),
      products = ("product", lambda x: ",".join(sorted(set(x))))
    )
    .reset_index()
  )
  return df
```