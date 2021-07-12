"""
Utils for finding ""green" documents


"""

from innovation_sweet_spots import PROJECT_DIR, logging
from innovation_sweet_spots.getters import crunchbase as cb
from innovation_sweet_spots.getters import gtr, misc
import innovation_sweet_spots.analysis.analysis_utils as iss
from innovation_sweet_spots.utils.text_cleaning_utils import clean_text
import innovation_sweet_spots.analysis.text_analysis as iss_text_analysis
import innovation_sweet_spots.utils.io as iss_io
from typing import Iterator

DATA_OUTPUTS = PROJECT_DIR / "outputs/data"

GREEN_TOPICS = [
    "Bioenergy",
    "Carbon Capture & Storage",
    "Climate & Climate Change",
    "Conservation Ecology",
    "Energy Efficiency",
    "Energy Storage",
    "Energy - Marine & Hydropower",
    "Energy - Nuclear",
    "Environmental Engineering",
    "Environmental economics",
    "Fuel Cell Technologies",
    "Solar Technology",
    "Sustainability Management",
    "Sustainable Energy Networks",
    "Sustainable Energy Vectors",
    "Waste Management",
    "Waste Minimisation",
    "Wind Power",
]

DEF_GREEN_GTR_PROJECTS = DATA_OUTPUTS / "gtr/green_gtr_project_ids.txt"
DEF_GREEN_CB_PROJECTS = DATA_OUTPUTS / "cb/green_cb_company_ids.txt"


def collect_green_keywords(
    save: bool = False,
    fpath=DATA_OUTPUTS / "aux/green_keywords_all.txt",
    keyword_sources: Iterator[str] = ["CB", "IK", "TechNav", "KK"],
) -> Iterator[str]:
    """
    Collects green keywords obtained from different sources, and combines those
    into one list.
    """

    keyword_paths = [
        misc.MISC_PATH / f"green_keywords_{s}.txt" for s in keyword_sources
    ]
    keyword_lists = [iss_io.read_list_of_terms(fpath) for fpath in keyword_paths]
    keywords = sorted(
        set([keyword for keyword_list in keyword_lists for keyword in keyword_list])
    )
    if save:
        iss_io.save_list_of_terms(keywords, DATA_OUTPUTS / fpath)
    return keywords


def get_chunks_from_tech_navigator(
    save=False, fpath=misc.MISC_PATH / "green_keywords_TechNav_before_review.txt"
) -> Iterator[str]:
    """NB: The list of keyword chunks requires subsequent manual revision"""
    # Relevant tech navigator columns
    cols = ["Technology Name", "Short Descriptor", "Brief description "]
    # Get tech navigator table
    tech_nav = misc.get_tech_navigator()
    # Get all chunks
    nlp = iss_text_analysis.setup_spacy_model()
    tech_chunks = []
    for col in cols:
        techs = tech_nav[col].to_list()
        techs = [s for s in techs if type(s) is str]
        tech_chunks += list(iss_text_analysis.chunk_forms(techs, nlp))
    tech_chunks_flat = sorted(set([t for ts in tech_chunks for t in ts]))
    if save:
        iss_io.save_list_of_terms(tech_chunks_flat, fpath)
    return tech_chunks_flat


def get_green_keywords(fpath=DATA_OUTPUTS / "aux/green_keywords_all.txt", clean=True):
    """Load the list of green keywords"""
    keywords = iss_io.read_list_of_terms(fpath)
    if clean:
        keywords = sorted(set([clean_text(keyword) for keyword in keywords]))
    return keywords


def find_green_gtr_projects_by_keywords(
    keywords: Iterator[str],
    min_mentions: int = 2,
    use_cached_documents: bool = True,
    docs=None,
) -> Iterator[str]:
    """Finds projects with the provided keywords"""
    # Load preprocessed project text documents
    if use_cached_documents:
        docs = gtr.get_cleaned_project_texts()
    green_ids = []
    # For each keyword, find all documents where the keyword is present
    logging.info(f"Searching {len(keywords)} key terms across {len(docs)} documents")
    for keyword in keywords:
        df = iss.is_term_present_in_sentences(
            keyword, docs.project_text.to_list(), min_mentions
        )
        green_ids.append(docs[df].project_id.to_list())
    # Flatten and deduplicate all ids associated with the keywords
    unique_green_ids = list(
        set([project_id for project_ids in green_ids for project_id in project_ids])
    )
    logging.info(
        f"Found {len(unique_green_ids)} documents that contain at least on of the keywords, mentioned at least {min_mentions} times"
    )
    return unique_green_ids


def find_green_gtr_projects_by_research_topic(research_topics=GREEN_TOPICS):
    gtr_projects = gtr.get_gtr_projects()
    gtr_topics = gtr.get_gtr_topics()
    link_gtr_topics = gtr.get_link_table("gtr_topic")
    gtr_project_topics = iss.link_gtr_projects_and_topics(
        gtr_projects, gtr_topics, link_gtr_topics
    )
    unique_green_ids = gtr_project_topics[
        gtr_project_topics.text.isin(research_topics)
    ].project_id.unique()
    logging.info(
        f"Found {len(unique_green_ids)} projects with the specified research topic tags"
    )
    return unique_green_ids


def find_green_gtr_projects(
    keywords,
    research_topics=GREEN_TOPICS,
    use_cached=True,
    fpath=DEF_GREEN_GTR_PROJECTS,
) -> pd.DataFrame:
    if not use_cached or (fpath.exists() is False):
        logging.info(f"Searching for green projects")
        ids_by_keywords = find_green_gtr_projects_by_keywords(keywords)
        ids_by_topics = find_green_gtr_projects_by_research_topic(research_topics)
        unique_green_ids = list(set(ids_by_keywords).union(set(ids_by_topics)))
        logging.info(f"Saving {len(unique_green_ids)} project IDs in {fpath}")
        iss_io.save_list_of_terms(unique_green_ids, fpath)
    else:
        unique_green_ids = iss_io.read_list_of_terms(fpath)
    logging.info(f"Found {len(unique_green_ids)} green projects")
    gtr_projects = gtr.get_gtr_projects()
    return gtr_projects[gtr_projects.project_id.isin(unique_green_ids)]


def cb_categories_for_group(cb_categories, group_name="Sustainability"):
    """Get unique cateogry group names"""
    df = cb_categories[-cb_categories.category_groups_list.isnull()]
    categories = sorted(
        set(df[df.category_groups_list.str.contains(group_name)].name.to_list())
    )
    return [s.lower() for s in categories]


def cb_category_groups(cb_categories):
    """Get unique cateogry group names"""
    split_groups = [
        split_string(s, separator=",")
        for s in cb_categories.category_groups_list.to_list()
    ]
    groups = sorted(set([group for groups in split_groups for group in groups]))
    return groups


def find_green_cb_companies(use_cached=True, fpath=DEF_GREEN_CB_PROJECTS):
    cb_orgs = cb.get_crunchbase_orgs_full()
    if not use_cached or (fpath.exists() is False):
        logging.info(f"Searching for green companies")
        # Find organisation categories
        cb_categories = cb.get_crunchbase_category_groups()
        cb_org_categories = cb.get_crunchbase_organizations_categories()
        # Find organisations and subgroups within the 'green' category
        green_tags = cb_categories_for_group(cb_categories, "Sustainability")
        green_orgs = cb_org_categories[cb_org_categories.category_name.isin(green_tags)]
        green_org_ids = list(green_orgs.organization_id.unique())
        logging.info(f"Saving {len(green_org_ids)} project IDs in {fpath}")
        iss_io.save_list_of_terms(green_org_ids, fpath)
    else:
        green_org_ids = iss_io.read_list_of_terms(fpath)
    green_orgs_table = cb_orgs[cb_orgs.id.isin(green_org_ids)]
    logging.info(f"Found {len(green_orgs_table)} green companies")
    return green_orgs_table
