#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:43:43 2021

@author: jdjumalieva
"""

import pytest
from innovation_sweet_spots.analysis.prototyping.public_discourse_analysis import \
    pd_data_collection_utils as dcu
    
    



def test_combine_articles():
    assert dcu.combine_articles([['a'], ['b', 'c']]) == ['a', 'b', 'c'], \
        "Should return flat list"


def test_filter_by_category():
    mock_articles = [
        {'id': 'artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'type': 'article',
         'sectionId': 'artanddesign',
         'sectionName': 'Art and design',
         'webPublicationDate': '2021-03-06T08:00:29Z',
         'webTitle': "Eco-homes become hot property in UK's zero-carbon ‘paradigm shift’",
         'webUrl': 'https://www.theguardian.com/artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'apiUrl': 'https://content.guardianapis.com/artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'fields': {'headline': "Eco-homes become hot property in UK's zero-carbon ‘paradigm shift’",
                    'trailText': 'Smart, low-carbon homes were once the preserve of one-off grand designs – now there are up to 30,000 projects in the pipeline<br>',
                    'body': '<p>Instead of parking spaces, it’s flowerbeds and vegetable planters that line the car-free street of Solar Avenue in Leeds.</p>'}},
        {'id': 'environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'type': 'article',
         'sectionId': 'environment',
         'sectionName': 'Environment',
         'webPublicationDate': '2020-03-27T21:00:38Z',
         'webTitle': 'UK government scraps green homes grant after six months',
         'webUrl': 'https://www.theguardian.com/environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'apiUrl': 'https://content.guardianapis.com/environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'fields': {'headline': 'UK government scraps green homes grant after six months',
                    'trailText': '£1.5bn scheme at heart of Boris Johnson’s ‘build back better’ promise has struggled since launch',
                    'body': '<p>The government has scrapped its flagship green homes grant scheme, just over six months after its launch.</p>'}}
        ]
    categories = ['Environment', 'Technology', 'Science']
    assert dcu.filter_by_category(mock_articles, categories) == [
        {'id': 'environment/2021/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'type': 'article',
         'sectionId': 'environment',
         'sectionName': 'Environment',
         'webPublicationDate': '2021-03-27T21:00:38Z',
         'webTitle': 'UK government scraps green homes grant after six months',
         'webUrl': 'https://www.theguardian.com/environment/2021/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'apiUrl': 'https://content.guardianapis.com/environment/2021/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'fields': {'headline': 'UK government scraps green homes grant after six months',
                    'trailText': '£1.5bn scheme at heart of Boris Johnson’s ‘build back better’ promise has struggled since launch',
                    'body': '<p>The government has scrapped its flagship green homes grant scheme, just over six months after its launch.</p>'}}
                                                                ]


def test_sort_by_year():
    mock_data = [
        {'id': 'artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'type': 'article',
         'sectionId': 'artanddesign',
         'sectionName': 'Art and design',
         'webPublicationDate': '2021-03-06T08:00:29Z',
         'webTitle': "Eco-homes become hot property in UK's zero-carbon ‘paradigm shift’",
         'webUrl': 'https://www.theguardian.com/artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'apiUrl': 'https://content.guardianapis.com/artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'fields': {'headline': "Eco-homes become hot property in UK's zero-carbon ‘paradigm shift’",
                    'trailText': 'Smart, low-carbon homes were once the preserve of one-off grand designs – now there are up to 30,000 projects in the pipeline<br>',
                    'body': '<p>Instead of parking spaces, it’s flowerbeds and vegetable planters that line the car-free street of Solar Avenue in Leeds.</p>'}},
        {'id': 'environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'type': 'article',
         'sectionId': 'environment',
         'sectionName': 'Environment',
         'webPublicationDate': '2020-03-27T21:00:38Z',
         'webTitle': 'UK government scraps green homes grant after six months',
         'webUrl': 'https://www.theguardian.com/environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'apiUrl': 'https://content.guardianapis.com/environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'fields': {'headline': 'UK government scraps green homes grant after six months',
                    'trailText': '£1.5bn scheme at heart of Boris Johnson’s ‘build back better’ promise has struggled since launch',
                    'body': '<p>The government has scrapped its flagship green homes grant scheme, just over six months after its launch.</p>'}}
        ]
    assert dcu.sort_by_year(mock_data) == defaultdict(list,
            {'2020':[{'id': 'environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'type': 'article',
         'sectionId': 'environment',
         'sectionName': 'Environment',
         'webPublicationDate': '2020-03-27T21:00:38Z',
         'webTitle': 'UK government scraps green homes grant after six months',
         'webUrl': 'https://www.theguardian.com/environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'apiUrl': 'https://content.guardianapis.com/environment/2020/mar/27/uk-government-scraps-green-homes-grant-after-six-months',
         'fields': {'headline': 'UK government scraps green homes grant after six months',
                    'trailText': '£1.5bn scheme at heart of Boris Johnson’s ‘build back better’ promise has struggled since launch',
                    'body': '<p>The government has scrapped its flagship green homes grant scheme, just over six months after its launch.</p>'}}
                     ]
             },
            {'2021': [{'id': 'artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'type': 'article',
         'sectionId': 'artanddesign',
         'sectionName': 'Art and design',
         'webPublicationDate': '2021-03-06T08:00:29Z',
         'webTitle': "Eco-homes become hot property in UK's zero-carbon ‘paradigm shift’",
         'webUrl': 'https://www.theguardian.com/artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'apiUrl': 'https://content.guardianapis.com/artanddesign/2021/mar/06/eco-homes-become-hot-property-in-uks-zero-carbon-paradigm-shift',
         'fields': {'headline': "Eco-homes become hot property in UK's zero-carbon ‘paradigm shift’",
                    'trailText': 'Smart, low-carbon homes were once the preserve of one-off grand designs – now there are up to 30,000 projects in the pipeline<br>',
                    'body': '<p>Instead of parking spaces, it’s flowerbeds and vegetable planters that line the car-free street of Solar Avenue in Leeds.</p>'}}
                      ]
             }
            )
             
def test_extract_text_from_html():
    mock_data = '<p>A dramatic growth in electric vehicles on Britain’s roads could see peak electricity demand jump by more than the capacity of the Hinkley Point C nuclear power station by 2030, according to National Grid.</p> <p>The number of plug-in cars and vans could reach 9m by 2030, up from <a href="https://www.theguardian.com/environment/2017/jul/08/electric-car-revolution-calculating-the-cost-of-green-motoring">around 90,000 today</a>, said the company, which runs the UK’s national transmission networks for electricity and gas.</p> <p>The impact of charging so many cars’ batteries would be to reverse the trend in recent years of falling electricity demand, <a href="https://www.theguardian.com/business/2017/mar/16/uk-climate-targets-will-raise-household-energy-bills-by-100-in-a-decade">driven by energy efficiency measures</a> such as better refrigerators and LED lighting.</p> <p>If electric vehicles were not <a href="https://www.theguardian.com/environment/2017/mar/20/electric-cars-uk-power-grids-charging-peaks-sse-demand-side-response">charged smartly</a> to avoid peaks and troughs in power demand, such as when people return home between 5pm and 6pm, peak demand could be as much as 8GW higher in 2030, National Grid said. </p> <p>Shifting the charging of cars to times when demand is lower would reduce the extra peak demand to 3.5GW, a smaller amount but still a similar capacity to the <a draggable="true" href="https://www.theguardian.com/business/2017/apr/21/hinkley-point-c-edf-somerset-nuclear-unions-brexit">new reactors being built at Hinkley Point in Somerset</a>.</p>'
    assert dcu.extract_text_from_html(mock_data, ['p', 'h2']) == ['A dramatic growth in electric vehicles on Britain’s roads could see peak electricity demand jump by more than the capacity of the Hinkley Point C nuclear power station by 2030, according to National Grid.', 'The number of plug-in cars and vans could reach 9m by 2030, up from around 90,000 today, said the company, which runs the UK’s national transmission networks for electricity and gas.', 'The impact of charging so many cars’ batteries would be to reverse the trend in recent years of falling electricity demand, driven by energy efficiency measures such as better refrigerators and LED lighting.', 'If electric vehicles were not charged smartly to avoid peaks and troughs in power demand, such as when people return home between 5pm and 6pm, peak demand could be as much as 8GW higher in 2030, National Grid said. ', 'Shifting the charging of cars to times when demand is lower would reduce the extra peak demand to 3.5GW, a smaller amount but still a similar capacity to the new reactors being built at Hinkley Point in Somerset.']