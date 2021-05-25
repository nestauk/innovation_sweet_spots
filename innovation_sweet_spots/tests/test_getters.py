from unittest import mock, TestCase
import unittest
from innovation_sweet_spots.getters.inputs import (
    get_gtr_projects,
    get_cb_data,
    build_projects,
    read_sql_table,
    PROJECT_DIR,
)
from pandas import DataFrame, read_csv, DataFrame
import os
from tempfile import NamedTemporaryFile
TEST_FILE = "test.csv"


class GetterTests(unittest.TestCase):
    @mock.patch("innovation_sweet_spots.getters.inputs.build_projects")
    def test_get_gtr_projects(self, mock_build_projects):
        data = [
            {"id": "X001", "title": "test_project_1"},
            {"id": "X002", "title": "test_project_2"},
        ]
        mock_build_projects.return_value = data
        # Test the case when use_cached=False
        with NamedTemporaryFile() as test_file:
            projects = get_gtr_projects(fpath=test_file.name, use_cached=False)
            assert type(projects) is list
            assert type(projects[0]) is dict
            assert len(projects) == len(data)
        # Test the case when use_cached=True
        projects = get_gtr_projects(fpath="non_existent_file.json")
        assert len(projects) == 0

    @mock.patch("innovation_sweet_spots.getters.inputs.read_sql_table")
    def test_get_cb_data(self, mock_read_sql_table):
        chunk_1 = DataFrame({"id": ["X001"], "title": ["test_project_1"]})
        chunk_2 = DataFrame({"id": ["X002"], "title": ["test_project_2"]})
        data = [chunk_1, chunk_2]
        mock_read_sql_table.return_value = data
        tables = get_cb_data(fpath="")
        assert type(tables) == dict
        for table_name in list(tables.keys()):
            table = tables[table_name]
            assert type(table) == dict
            assert type(table["path"]) == str
            assert type(table["data"]) == DataFrame
            assert read_csv(table["path"]).equals(table["data"])
            os.remove(table["path"])
