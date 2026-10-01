import logging
import extract
import transform
import load
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("myapp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    pipeline_start = time.time()
    logger.info('ETL pipeline started')

    raw_data, extract_metrics = extract.extract()
    logger.info("Extracted %s source tables", len(extract_metrics))

    dw_tables, transform_metrics = transform.transform(raw_data)
    logger.info("Transformation metrics: %s", transform_metrics)

    load_metrics = load.load(dw_tables)
    logger.info("Load metrics: %s", load_metrics)

    pipeline_duration = round(time.time() - pipeline_start, 2)
    logger.info("ETL pipeline finished in %s seconds", pipeline_duration)

if __name__ == '__main__':
    main()