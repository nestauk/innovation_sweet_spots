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
        with NamedTemporaryFile() as test_file:
            # Test the case when use_cached=False
            projects = get_gtr_projects(fpath=test_file.name, use_cached=False)
            assert type(projects) is list
            assert type(projects[0]) is dict
            assert len(projects) == len(data)
            # Test the case when use_cached=True
            projects_from_cache = get_gtr_projects(
                fpath=test_file.name, use_cached=True
            )
            assert projects_from_cache == projects
        # Test the case when use_cached=True but file does not exist
        non_existent_file = "non_existent_file.json"
        assert os.path.exists(non_existent_file) is False
        projects = get_gtr_projects(fpath=non_existent_file)
        assert os.path.exists(non_existent_file)
        os.remove(non_existent_file)

    @mock.patch("innovation_sweet_spots.getters.inputs.safe_load")
    @mock.patch("innovation_sweet_spots.getters.inputs.read_sql_table")
    def test_get_cb_data(self, mock_read_sql_table, mock_safe_load):
        chunk_1 = DataFrame({"id": ["X001"], "title": ["test_project_1"]})
        chunk_2 = DataFrame({"id": ["X002"], "title": ["test_project_2"]})
        data = [chunk_1, chunk_2]
        mock_read_sql_table.return_value = data
        # Mock table specification
        test_table_name = "test"
        mock_safe_load.return_value = {test_table_name: ["field_1", "field_2"]}
        # Test the case when use_cached=False
        tables = get_cb_data(fpath=PROJECT_DIR, use_cached=False)
        assert type(tables) is list
        for table in tables:
            assert type(table) is dict
            assert type(table["name"]) is str
            assert type(table["path"]) is str
            assert type(table["data"]) is DataFrame
            assert table["name"] == test_table_name
            assert read_csv(table["path"]).equals(table["data"])
        # Test the case when use_cached=True
        tables_reloaded = get_cb_data(fpath=PROJECT_DIR, use_cached=True)
        assert tables_reloaded[0]["data"].equals(tables[0]["data"])
        # Test the case when use_cached=True, but file does not exist
        tables_reloaded = get_cb_data(fpath="non_existent_folder")
        for table in tables_reloaded:
            assert len(table["data"]) == 0

    def tearDown(self):
        if os.path.exists(TEST_FILE):
            os.remove(TEST_FILE)
