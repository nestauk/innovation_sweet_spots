from unittest import mock, TestCase
import unittest
from innovation_sweet_spots.getters.inputs import get_gtr_projects, build_projects
from tempfile import NamedTemporaryFile


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
