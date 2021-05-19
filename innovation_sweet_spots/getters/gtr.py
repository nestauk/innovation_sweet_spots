from data_getters.gtr import build_projects
from data_getters.inspector import get_schemas
import pandas as pd

config_path = "/Users/karliskanders/Documents/_coding/AWS/mysqldb_team.config"
schemas = get_schemas(config_path)

# print(schemas['gtr'].keys())
# for p in list(schemas['gtr']['gtr_projects'].columns):
#     print(p)
# fields = ["id", "title", "start", "end", "abstractText", "techAbstractText"]
fields = ["id", "title"]

projects = build_projects(
    config_path, chunksize=5000, table_wildcards=["gtr_projects"], desired_fields=fields
)
df = pd.DataFrame(projects)
df.to_csv("gtr_projects.csv")
