import logging
import extract
import transform
import load

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    logger.info('Started')
    extract.extract()
    transform.transform()
    load.load()
    logger.info('Finished')

if __name__ == '__main__':
    main()