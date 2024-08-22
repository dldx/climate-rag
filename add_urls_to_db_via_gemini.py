from tools import add_urls_to_db, get_vector_store
import logging
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from rich.progress import Progress

logging.basicConfig(level=logging.INFO)
db = get_vector_store()
urls = """file:///home/durand/Downloads/1723805577_Snapshot_of_Indias_Oil_Gas_Data-July2024_A5.pdf
https://ppac.gov.in/download.php?file=importantnews/1724215460_crude_en.pdf
https://ppac.gov.in/download.php?file=importantnews/1724213587_PP_9_a_DailyPriceMSHSD_Metro_21.08.2024.pdf
https://ppac.gov.in/download.php?file=importantnews/1724213503_PP_9_a_DailyPriceMSHSD_Metro_21.08.2024.pdf
https://ppac.gov.in/download.php?file=importantnews/1722410339_domestic-gas-price-august-2024.pdf
https://ppac.gov.in/download.php?file=importantnews/1711944273_gas-ceiling.pdf
https://ppac.gov.in/download.php?file=rep_studies/1716870573_LPG_Profile_Report_FY_2023-24-Web.pdf
https://ppac.gov.in/download.php?file=rep_studies/1721728196_Final-Monthly-Report.pdf
https://ppac.gov.in/download.php?file=rep_studies/1721800076_Monthly-Gas-Report-Jun24-WebV.pdf
https://ppac.gov.in/download.php?file=rep_studies/1724132295_LPG_Profile_Report_Q1
https://ppac.gov.in/download.php?file=govtnotlaws/1706765053_gazetter_on_ms_hsd_control_order_retailing.pdf
https://ppac.gov.in/download.php?file=govtnotlaws/1706764878_sale_of_bio_diesel_policyr_apr_2019.pdf
https://ppac.gov.in/download.php?file=govtnotlaws/1714040983_Petroleum_Natural_Gas_Amendment_Rules_2018.pdf
https://ppac.gov.in/download.php?file=govtnotlaws/1706676335_National_Bio_Fuel_Policy_June_2018.pdf
https://ppac.gov.in/download.php?file=govtnotlaws/1715943099_The_Sexual_Harassment_of_Women_at_Workplace_Act_2013.pdf
https://ppac.gov.in/download.php?file=govtnotlaws/1706765156_the_petroleum_2006.pdf
https://ppac.gov.in/download.php?file=govtnotlaws/1706764752_oil_industry(developement)_act_1974.pdf
https://ppac.gov.in/download.php?file=govtnotlaws/1706683201_Petroleum_and_Minerals_pipeline_Act_1962.pdf
https://ppac.gov.in/download.php?file=circularnotices/1715944073_ICC_under_POSH_Act.pdf
https://ppac.gov.in/download.php?file=whatsnew/1724132295_LPG_Profile_Report_Q1
https://ppac.gov.in/download.php?file=whatsnew/1723815267_ICR_July_2024_16082024_Final.pdf
https://ppac.gov.in/download.php?file=rep_studies/1723805577_Snapshot_of_Indias_Oil_Gas_Data-July2024_A5.pdf
https://ppac.gov.in/download.php?file=rep_studies/1723815267_ICR_July_2024_16082024_Final.pdf
https://ppac.gov.in/download.php?file=rep_studies/1719411262_Final_Book_June%202024_Final_for_upload.pdf"""
urls = urls.split("\n")

partial_func = partial(add_urls_to_db, db=db, use_gemini=True)

def process_url(url, progress, task_id):
    result = partial_func([url])
    progress.update(task_id, advance=1)
    return result

with Progress() as progress:
    task_id = progress.add_task("[green]Processing URLs...", total=len(urls))
    with Pool(4) as pool:
        pool.starmap(process_url, [(url, progress, task_id) for url in urls])
