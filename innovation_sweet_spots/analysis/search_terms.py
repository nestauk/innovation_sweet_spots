"""
Definitions of search terms

"""
# Relevant GTR categories
categories_to_gtr_categories = {
    "Batteries": ["Energy Storage"],
    "Bioenergy": ["Bioenergy"],
    "Carbon Capture & Storage": ["Carbon Capture & Storage"],
    "Demand management": None,
    "EV": None,
    "Heating & Buildings": None,
    "Hydrogen & Fuel Cells": ["Fuel Cell Technologies"],
    "Nuclear": ["Energy - Nuclear"],
    "Solar": ["Solar Technology"],
    "Wind & Offshore": ["Wind Power", "Energy - Marine & Hydropower"],
}

# Relevant CB categories
categories_to_cb_categories = {
    "Batteries": ["battery", "energy storage"],
    "Bioenergy": ["biomass energy", "biofuel"],
    "Carbon Capture & Storage": None,
    "Demand management": [
        "energy management",
        "smart building",
        "smart cities",
        "smart home",
    ],
    "EV": ["electric vehicle"],
    "Heating & Buildings": ["green building"],
    "Heating (all other)": None,
    "Hydrogen & Fuel Cells": ["fuel cell"],
    "Nuclear": ["nuclear"],
    "Solar": ["solar"],
    "Wind & Offshore": ["wind energy"],
}

categories_to_lda_topics = {
    "Heat pumps": [19],
    "Waste heat": [19],
    "Geothermal energy": [19],
    "Solar thermal": [19],
    "Heat storage": [19],
    "District heating": [19],
    "Electric boilers": [19],
    "Micro CHP": [19],
    "Hydrogen heating": [19, 94],
    "Biomass heating": [19, 68, 92, 93],
    "Building insulation": [149],
    "Radiators": [149],
    "Energy management": [138],
}

reference_categories_to_lda_topics = {
    "Solar": [2],
    "Wind & Offshore": [67],
    "Batteries": [40],
    "Hydrogen & Fuel Cells": [94, 130],
    "Bioenergy": [68, 92, 93],
    "Carbon Capture & Storage": [39],
    "Heating (all other)": [19],
}

categories_keyphrases = {
    # Heating subcategories
    "Heat pumps": [["heat pump"]],
    "Geothermal energy": [["geothermal", "energy"], ["geotermal", "heat"]],
    "Solar thermal": [["solar thermal"]],
    "Waste heat": [["waste heat"], ["heat recovery"]],
    "Heat storage": [
        ["heat stor"],
        ["thermal energy stor"],
        ["thermal stor"],
        ["heat batter"],
    ],
    "District heating": [["heat network"], ["district heat"]],
    "Electric boilers": [["electric", "boiler"], ["electric heat"]],
    "Micro CHP": [["combined heat power", "micro"], ["micro", "chp"], ["mchp"]],
    "Hydrogen heating": [
        ["hydrogen", "boiler"],
        ["hydrogen", "heating"],
        ["hydrogen heat"],
        # ['green hydrogen']
    ],
    "Biomass heating": [
        ["biomass", "heating"],
        ["biomass", " heat"],
        ["pellet", "boiler"],
        ["biomass", "boiler"],
    ],
    # Building insulation, retrofit & energy management
    "Building insulation": [
        ["insulat", "build"],
        ["insulat", "hous"],
        ["insulat", "home"],
        ["insulat", "retrofit"],
        ["cladding", "hous"],
        ["cladding", "build"],
        ["glazing", "window"],
        ["glazed", "window"],
    ],
    "Radiators": [["radiator"]],
    "Energy management": [
        ["smart home", "heat"],
        ["demand response", "heat"],
        ["load shift", "heat"],
        ["energy management", "build"],
        ["energy management", "domestic"],
        ["energy management", "hous"],
        ["energy management", "home"],
        ["thermostat"],
        ["smart meter"],
    ],
}

reference_category_keyphrases = {
    "Batteries": [["battery"], ["batteries"]],
    "Solar": [
        ["solar energy"],
        ["solar", "renewable"],
        ["solar cell"],
        ["solar panel"],
        ["photovoltaic"],
        ["perovskite"],
        ["pv cell"],
    ],
    "Carbon Capture & Storage": [
        [" ccs "],
        ["carbon storage"],
        ["carbon capture"],
        ["co2 capture"],
        ["carbon abatement"],
    ],
    "Bioenergy": [
        ["biofuel"],
        ["ethanol", "fuel"],
        ["butanol", "fuel"],
        ["syngas"],
        ["biogas"],
        ["biochar"],
        ["biodiesel"],
        ["torrefaction", "biomass"],
        ["pyrolisis", "biomass"],
    ],
    "Hydrogen & Fuel Cells": [[" hydrogen "], ["fuel cell"], ["sofc"]],
    "Wind & Offshore": [
        ["wind energy"],
        ["wind", "renewable"],
        ["wind turbine"],
        ["wind farm"],
        ["wind generator"],
        ["wind power"],
        ["wave", "renewable"],
        ["tidal energy"],
        ["tidal turbine"],
        ["offshore energy"],
        ["offshore wind"],
        ["onshore wind"],
    ],
    "Heating (all other)": [
        ["infrared", "heating"],
        ["heat", "build"],
        ["heat", "home"],
        ["heat", "hous"],
        ["heat", "domestic"],
        ["heat", "renewable"],
        ["heating"],
    ],
}
