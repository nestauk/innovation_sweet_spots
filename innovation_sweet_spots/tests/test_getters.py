from unittest import mock, TestCase
import unittest
from innovation_sweet_spots.getters.inputs import get_gtr_projects, build_projects
from pandas import DataFrame, read_csv
import os

TEST_FILE = "test.csv"


class GetterTests(unittest.TestCase):
    @mock.patch("innovation_sweet_spots.getters.inputs.build_projects")
    def test_get_gtr_projects(self, mock_build_projects):
        data = [
            {"id": "X001", "title": "test_project_1"},
            {"id": "X002", "title": "test_project_2"},
        ]
        mock_build_projects.return_value = data
        df = get_gtr_projects(fpath=TEST_FILE)
        assert type(df) == DataFrame
        assert len(df) == len(data)
        assert read_csv(TEST_FILE).equals(df)
        os.remove(TEST_FILE)


if __name__ == "__main__":
    unittest.main()
