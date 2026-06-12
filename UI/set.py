from ucimlrepo import fetch_ucirepo
import pandas as pd
d = fetch_ucirepo(id=602)
df = pd.concat([d.data.features, d.data.targets], axis=1)
df.to_csv("dry_bean.csv", index=False)