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
        # Test the case when use_cached=True, but file does not exist
        projects = get_gtr_projects(fpath="non_existent_file.json")
        assert len(projects) == 0

    @mock.patch("innovation_sweet_spots.getters.inputs.safe_load")
    @mock.patch("innovation_sweet_spots.getters.inputs.read_sql_table")
    def test_get_cb_data(self, mock_read_sql_table, mock_safe_load):
        chunk_1 = DataFrame({"id": ["X001"], "title": ["test_project_1"]})
        chunk_2 = DataFrame({"id": ["X002"], "title": ["test_project_2"]})
        data = [chunk_1, chunk_2]
        mock_read_sql_table.return_value = data
        with NamedTemporaryFile() as test_file:
            # Mock table specification
            mock_safe_load.return_value = {test_file.name: ["field_1", "field_2"]}
            # Test the case when use_cached=False
            tables = get_cb_data(fpath="", use_cached=False)
            assert type(tables) == list
            for table in tables:
                assert type(table) == dict
                assert type(table["name"]) == str
                assert type(table["path"]) == str
                assert type(table["data"]) == DataFrame
                assert table["name"] == test_file.name
                assert read_csv(table["path"]).equals(table["data"])
            # Test the case when use_cached=True
            tables_reloaded = get_cb_data(fpath="", use_cached=True)
            assert tables_reloaded[0]["data"].equals(tables[0]["data"])
        # Test the case when use_cached=True, but file does not exist
        tables_reloaded = get_cb_data(fpath="non_existent_folder")
        for table in tables_reloaded:
            assert len(table["data"]) == 0
