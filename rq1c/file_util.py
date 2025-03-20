import os

def initlog(xp_path, train_method):
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_name = f"{train_method}"
    file_handler = logging.FileHandler(os.path.join(xp_path, logger_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


import logging


def logger_config(xp_path, logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.INFO)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(os.path.join(xp_path, "log.txt"), encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


if __name__ == "__main__":
    logger = logger_config(log_path='log.txt', logging_name='据说名字长一点容易被人记住')
    logger.info("info")
    logger.error("error")
    logger.debug("debug")
    logger.warning("warning")
    print('print和logger输出是有差别的！')
