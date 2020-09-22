import logging

LOG_FILENAME = '/home/user10110/code/error.log'

def print_log():
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename=LOG_FILENAME,
                        filemode='w',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志,a是追加模式,默认a
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'  # 日志格式
                        )

    logging.info('info')
    logging.warning('warning')
    logging.error('error')
    logging.critical('critial')



