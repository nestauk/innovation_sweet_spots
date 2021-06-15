import pytest
from unittest import mock
from tempfile import NamedTemporaryFile
import pandas as pd
from pathlib import Path

from innovation_sweet_spots.getters.gtr import pullout_gtr_links


@mock.patch("innovation_sweet_spots.getters.gtr.get_path_to_specific_link_table")
@mock.patch("innovation_sweet_spots.getters.gtr.get_link_table")
def test_pullout_gtr_links(mock_get_link_table, mock_get_path_to_specific_link_table):
    with NamedTemporaryFile() as test_file:
        mock_table = pd.DataFrame(
            data={"table_name": ["table_a", "table_b", "table_b", "table_c"]}
        )
        mock_get_link_table.return_value = mock_table
        mock_get_path_to_specific_link_table.return_value = Path(test_file.name)

        # Test the case when only one table is specified
        pullout_table_name = ["table_b"]
        pullout_gtr_links(pullout_table_name)
        df = pd.read_csv(test_file.name)
        assert len(df) == 2
        table_names = list(df.table_name.unique())
        assert len(table_names) == 1
        assert table_names[0] == pullout_table_name[0]
        assert df.equals(
            mock_table[mock_table.table_name == pullout_table_name[0]].reset_index(
                drop=True
            )
        )

        # Test the case when more than one table is specified
        # (Note that the test will overwrite the tables as they'll have the same name)
        pullout_table_name = ["table_b", "table_c"]
        pullout_gtr_links(pullout_table_name)
        df = pd.read_csv(test_file.name)
        assert len(df) == 1
        table_names = list(df.table_name.unique())
        assert len(table_names) == 1
        assert table_names[0] == pullout_table_name[1]
