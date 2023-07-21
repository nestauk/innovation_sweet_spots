"""
Script to label the childcare companies using ChatGPT

Usage:

"""
from innovation_sweet_spots.analysis.notebooks.y2023_childcare import utils
from innovation_sweet_spots import PROJECT_DIR, logging
import innovation_sweet_spots.getters.google_sheets as gs
from innovation_sweet_spots.getters.preprocessed import (
    get_preprocessed_crunchbase_descriptions,
)
import openai
import innovation_sweet_spots.utils.openai as openai_utils

openai.api_key = openai_utils.get_api_key()
import pandas as pd
import typer
import csv

# Name of the session
SESSION = "v2023_03_14"
# Companies to label
CB_IDS_PATH = (
    PROJECT_DIR / f"outputs/2023_childcare/interim/openai/cb_ids_{SESSION}.csv"
)
# Output table
OUTPUT_FILE = (
    PROJECT_DIR / f"outputs/2023_childcare/interim/openai/chatgpt_labels_{SESSION}.csv"
)
# Fields/columns of the outputs
FIELDS = [
    "id",
    "object",
    "created",
    "model",
    "usage",
    "choices",
    "cb_id",
]

PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant who is labelling companies by using predefined categories.",
    },
    {
        "role": "user",
        "content": "'You are a helpful assistant who is labelling companies by using predefined categories. This is for a project to map companies working on improving childcare, parental support and early years education solutions, focussed on children between 0-5 years old. Your are given keywords for each category, and the company description. You will output one or maximum two categories that best match the company description. You can also label the company as “Not relevant”. For example, we are not interested in solutions for middle or high schoolers; universities; healthcare; or companies not related to families or education.\n\nHere are the categories and their keywords provided in the format Category name - keywords.\nContent: General - curriculum, education content, resource\nContent: Numeracy - numeracy, mathematics, coding\nContent: Literacy - phonics, literacy, reading, ebook\nContent: Play - games, play, toys\nContent: Creative - singing, song, songs, art, arts, drawing, painting\nTraditional models: Preschool - pre school, kindergarten, montessori\nTraditional models: Child care - child care, nursery, child minder, babysitting\nTraditional models: Special needs - special needs, autism, mental health\nManagement - management, classroom technology, monitoring technology, analytics, waitlists\nTech - robotics, artificial intelligence, machine learning, simulation\nWorkforce: Recruitment - recruitment, talent acquisition, hiring\nWorkforce: Training - teacher training, skills\nWorkforce: Optimisation - retention, wellness, shift work\nFamily support: General - parents, parenting advice, nutrition, feeding, sleep, travel, transport\nFamily support: Peers - social network, peer to peer\nFamily support: Finances - finances, cash, budgeting.\n\nHere are examples of company descriptions and categories.\n\nExample 1: Description: privacy- first speech recognition software delivers voice- enabled experiences for kids of all ages, accents, and dialects. has developed child- specific speech technology that creates highly accurate, age- appropriate and safe voice- enabled experiences for children. technology is integrated across a range of application areas including toys, gaming, robotics, as well as reading and English Language Learning . Technology is fully and GDPR compliant- offering deep learning speech recognition based online and offline embedded solutions in multiple languages. Industries: audio, digital media, events\nCategory: <Tech>\n\nExample 2: Description: is a personalized learning application to improve math skills. is a personalized learning application to improve math skills. It works by identifying a child’s level, strengths and weaknesses, and gradually progressing them at the rate that’s right for them. The application is available for download on the App Store and Google Play. Industries: accounting, finance, financial services.\nCategory: <Content: Numeracy>\n\nNow categorise this company: Description: The company helps over 1.8M middle-school, high-school and college students worldwide, to understand and solve their math problems step-by-step.",
    },
    {"role": "assistant", "content": "<Not relevant>"},
    {
        "role": "user",
        "content": "Description: The company  is an EdTech startup company providing game-based math and reading courses to students in pre-kindergarten to grade five.",
    },
    {"role": "assistant", "content": "<Content: Numeracy> and <Content: Literacy>"},
    {
        "role": "user",
        "content": "Description: The company is a global digital- first entertainment company for kids. The company is a global entertainment company that creates and distributes inspiring and engaging stories to expand kids’ worlds and minds. Founded in 2018, with offices in and, The company creates, produces and publishes thousands of minutes of video and audio content every month with the goal of teaching compassion, empathy and resilience to kids around the world.",
    },
    {"role": "assistant", "content": "<Content: General>"},
]


def run_chatgpt_labelling(
    test: bool = False,
    n_test: int = 3,
):
    """
    Runs a session to label companies according to our taxonomy

    Args:
        test (bool): If True, runs a test session with a small number of companies
        n_test (int): Number of companies to label in test mode
    """
    # Fetch list of companies we will label in this session
    if test:
        cb_ids_to_check = (
            pd.read_csv(CB_IDS_PATH).cb_id.sample(n_test, random_state=42).to_list()
        )
        logging.info(f"TEST SESSION: Labelling {n_test} companies")
    else:
        cb_ids_to_check = pd.read_csv(CB_IDS_PATH).cb_id.to_list()
        logging.info(
            f"Number of companies to label in this session: {len(cb_ids_to_check)}"
        )
    n_ids_total = len(cb_ids_to_check)
    # Fetch table with labelled companies and cross-check with the cb_id list
    if OUTPUT_FILE.exists():
        # Load table with labelled companies
        outputs_df = pd.read_csv(OUTPUT_FILE, names=FIELDS, header=None)
        # Remove companies that are already labelled
        cb_ids_to_check = list(set(cb_ids_to_check) - set(outputs_df.cb_id.to_list()))
        logging.info(f"{n_ids_total-len(cb_ids_to_check)} companies already labelled")
        logging.info(
            f"Number of companies to label in this run: {len(cb_ids_to_check)}"
        )

    # Get longlist of companies
    longlist_df = gs.download_google_sheet(
        google_sheet_id=utils.AFS_GOOGLE_SHEET_ID,
        wks_name="list_v2",
    )
    # Load a table with processed company descriptions
    processed_texts = get_preprocessed_crunchbase_descriptions()
    # Add processed texts to longlist
    company_texts = (
        longlist_df[["cb_id", "company_name", "relevant", "model_relevant"]]
        .merge(processed_texts.rename(columns={"id": "cb_id"}), on="cb_id", how="left")
        .query("cb_id in @cb_ids_to_check")
    )

    # Prepare queries by adding company descriptions to the prompt
    queries = {}
    for i, row in company_texts.iterrows():
        queries[row["cb_id"]] = PROMPT + [
            {"role": "user", "content": f"Description: {row['description']}"}
        ]

    # Print prompt
    openai_utils.print_prompt(PROMPT)

    # Stream outputs to the output csv table
    with open(OUTPUT_FILE, "a") as output_file:
        writer = csv.DictWriter(output_file, FIELDS)
        for query in queries:
            chatgpt_output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=queries[query],
                temperature=0.5,
                max_tokens=1000,
            ).to_dict()
            chatgpt_output.update({"cb_id": query})
            writer.writerow(chatgpt_output)
            # Print output
            try:
                print(
                    f"company_name: {company_texts.query('cb_id == @query')['company_name'].values[0]}"
                )
                print(f"user: {queries[query][-1]['content']}")
                print(
                    f"assistant: {chatgpt_output['choices'][0]['message']['content']}"
                )
                print("-------")
            except:
                print(chatgpt_output)


if __name__ == "__main__":
    typer.run(run_chatgpt_labelling)
