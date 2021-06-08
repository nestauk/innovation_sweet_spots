"""Metaflow to fetch data from DAPS1 (`nestauk/nesta`).

GtR and Crunchbase data pipeline currently reside `nestauk/nesta`.
fetched using the `nestauk/data_getters` internal library.
"""
import os

from metaflow import conda_base, FlowSpec, Parameter, step

ENV_VAR = "MYSQL_CONFIG"

LIBRARIES = {"pymysql": "0.9.3", "sqlalchemy": "1.3.4", "pandas": ">1"}


@conda_base(python="3.7", libraries=LIBRARIES)
class CreatechNestaGetter(FlowSpec):

    db_config_path = Parameter("db-config-path", type=str, default=os.environ[ENV_VAR])

    @step
    def start(self):
        self.next(self.fetch_names)

    @step
    def fetch_names(self):
        """Fetch Organisation (GtR & crunchbase) names."""
        from daps1_utils import get_engine
        from gtr_utils import get_names as get_gtr_names
        from cb_utils import get_uk_names as get_cb_names

        if self.db_config_path is None:
            raise ValueError(
                f"`db_config_path` was not set. Pass in a config path as a "
                f"flow argument or set {ENV_VAR} env variable."
            )

        engine = get_engine(self.db_config_path)
        con = engine.connect()

        self.gtr_names = get_gtr_names(con)
        self.crunchbase_names = get_cb_names(con)

        self.next(self.fetch_cb)

    @step
    def fetch_cb(self):
        """Fetch Crunchbase tables."""
        from cb_utils import CB_PATH, fetch_save_crunchbase

        if os.path.exists(CB_PATH) is False:
            os.makedirs(CB_PATH)

        fetch_save_crunchbase()

        self.next(self.fetch_gtr)

    @step
    def fetch_gtr(self):
        """Fetch GtR tables."""
        from gtr_utils import GTR_PATH, fetch_save_gtr_tables

        if os.path.exists(GTR_PATH) is False:
            os.mkdir(GTR_PATH)

        fetch_save_gtr_tables()

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    CreatechNestaGetter()
