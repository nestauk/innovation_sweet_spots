from unittest import mock, TestCase
import unittest
from innovation_sweet_spots.getters.inputs import get_gtr_projects, build_projects
from tempfile import NamedTemporaryFile
import os


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
