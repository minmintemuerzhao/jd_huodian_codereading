import logging
import os

from functools import partial
import numpy as np
import pandas as pd

# from databricks.koalas.utils import default_session
from heos.boundary import Boundary
from heos.config import load_config
from heos.processor import data_frame, _re_sample_s, load_sensor
from heos.tools import date_format_java, read_csv, mean, median, to_unixtime, read_parquet, to_parquet, merge, \
    any_exists, all_exists, csv_to_parquet, init_spark_local, do_patch_spark, join_path
from pandas import DataFrame as pDataFrame
from pyspark.sql import functions as fn
from pyspark.sql import Window


def main(
        do_merge=False,
        do_convert=False,
        do_process_coal_quality=False,
        do_describe=False,
        do_check_boundary=False,
        do_re_sample=False,
        do_batch_warning=False,
        do_process_feature=False,
        do_process_coal=False,
        do_concat=False,
        do_extra=False,
        do_output=False,
        do_output_point=False
):
    config = load_config("config.yml")

    '''
    not do_merge 是一个
    do_merge:在古交电厂三号机数据中。是按照时间划分成小文件的，需要按照点表的名字进行合并成一个文件
    '''
    # 将split文件夹里面的文件，按照前缀的名字进行merge，写到merge的路径下。
    # 没看懂的问题是：1、代码中没看出来，merge里面的文件是csv格式的，
    #         2、单纯看服务器上的文件，spliet和merge的文件名对不上
    #         3、从merge开始，数据没有csv_schema，parquet_schema等里面的row_number了
    not do_merge or merge_splits(
        source=config.get("paths").get("splits"),
        target=config.get("paths").get("merge")
    )

    # 将merge中的csv转化成parquet存到convert里面
    not do_convert or _init_spark() and convert_csv_to_parquet(
        source=config.get("paths").get("merge"),
        target=config.get("paths").get("convert"),
        schema=config.get("spark").get("csv_schema")
    )
    # 根据coal_quality_config.csv的配置信息，将coal_quality中的煤炭数据分别存成几个parquet数据到convert中去
    # 此处是：收到基水份,收到基灰份,干燥无灰基挥发份,低位发热量,飞灰可燃物含量,大渣可燃物含量这六个指标的值
    # 问题是：次处生成的六个文件，不确定是不是parquet格式的（按照代码应该是，但是服务器里的是不是？）
    not do_process_coal_quality or _init_spark() and process_coal_quality(
        csv="coal_quality_config.csv",
        source=config.get("paths").get("coal_quality"),
        target=config.get("paths").get("convert")
    )
    # 根据key分组，并统计最大值，最小值，均值
    # 问题：目的是什么，这个函数没太看明白
    not do_describe or _init_spark() and describe(
        source=config.get("paths").get("convert") + "/*",
        target=config.get("paths").get("describe"),
        schema=config.get("spark").get("parquet_schema"),
        key="point"
    )
    # re_sample_all其实就是把convert里面的数据，根据up_time以及boundary等的要求，
    # 采用多线程，对每个文件进行处理，得到对应时间段的处理后的数据
    # 注意：conver中有126个文件，而re_sample有803个文件，其中conver中的文件会一一对应生成放到re_sample中，
    not do_re_sample or _init_spark() and re_sample_all(
        source=config.get("paths").get("convert"),
        target=config.get("paths").get("re_sample"),
        boundary=Boundary("min_max.csv"),
        schema=config.get("spark").get("parquet_schema"),
        up_time=config.get("up_time")
    )
    # 检查boundary
    # 问题：1、check_boundary函數中和data_all_alpha_v3.0.csv的列的个数好像对不上
    #       2、为什么一行会有多个kks，而每一行对应的最大最小值必须唯一
    not do_check_boundary or check_boundary(
        boundary=Boundary("min_max.csv"),
        csv="data_all_alpha_v3.0.csv"
    )
    # 过滤并取出符合最大最小值范围的值
    # 问题：1、df.value，这个语法应该是df.values，之前代码是可以跑通的么？
    #       2、batch_warning对应文件夹下是空的
    #       3、这步的目的是什么，是符合取值范围的留下，不符合的呢？怎么处理，看函数没看明白
    not do_batch_warning or batch_waning(
        source=config.get("paths").get("re_sample"),
        target=config.get("paths").get("batch_warning"),
        csv="batch_warning.csv"
    )
    # 这步的目的是什么，没太看明白。
    not do_process_feature or process_feature(
        source=config.get("paths").get("re_sample"),
        source2=config.get("paths").get("batch_warning"),
        target=config.get("paths").get("feature"),
        #csv="data_all_alpha_soot_v1.0.csv"
        csv="data_all_alpha_v3.0.csv"
    )
    # 从coal_hdf中取出来data_all_alpha_v3.0.csv中的CoalQuality对应的六个特征指标的数据，并存到feature路径下
    # yml里面并没有coal_hdf的路径，
    not do_process_coal or process_coal(
        target=config.get("paths").get("feature"),
        csv="data_all_alpha_v3.0.csv",
        coal_hdf=config.get("paths").get("coal_hdf")
    )
    # 进行concat操作，把feature中分开存的东西都拼接写到concat里面去
    not do_concat or concat_sensors(
        source=config.get("paths").get("feature"),
        target=config.get("paths").get("concat"),
        #csv="data_all_alpha_soot_v1.0.csv"
        csv="data_all_alpha_v3.0.csv"
    )
    # 根据机组实际负荷和各个给煤机瞬时给煤量反馈，拟合并计算额外给煤量
    not do_extra or calculate_extra(
        source=config.get("paths").get("concat"),
        target=config.get("paths").get("extra")
    )
    # 对各种值进行判断处理，获取输出结果
    not do_output or output(
        #source=config.get("paths").get("concat"),
        source=config.get("paths").get("extra"),
        #csv="data_all_alpha_soot_v1.0.csv",
        target=config.get("paths").get("result"),
        csv="data_all_alpha_v3.0.csv"
    )
    # 把source中的数据，以parquet和csv的格式存到result中去
    not do_output_point or output_point(
        source=config.get("paths").get("re_sample"),
        target=config.get("paths").get("result"),
        csv="data_all_point_v1.csv"
    )


def output_point(source, target, csv):
    df = read_csv(csv, sep=",", usecols=[0, 1])
    result = None
    # assert all_exists([os.path.join(source, str(rn)) for rn in range(len(df))])
    for rn, row in df.iterrows():
        _, chn_name = row
        sensor = read_parquet(os.path.join(source, chn_name), metadata=_date_value_frame)
        if result is None:
            result = sensor
            result.columns = [chn_name]
        else:
            # noinspection PyUnresolvedReferences
            result[chn_name] = sensor
    result['date'] = result.index
    result.round(4).to_parquet(target)
    result.round(4).to_csv(target + ".csv", index=False)


def check_boundary(csv, boundary):
    df = read_csv(csv, sep="\\s+")
    for rn, row in df.iterrows():
        kks_names, chn_name, flag = row
        _logger.info("checking boundary for {0}".format(chn_name))
        points = kks_names.split(",")
        if len(points) == 1:
            continue
        else:
            boundaries = [boundary.get(point) for point in points]
            min_set = {b[0] for b in boundaries}
            max_set = {b[1] for b in boundaries}
            if len(min_set) != 1:
                _logger.error("min value is not unique for group {0}".format(chn_name))
            if len(max_set) != 1:
                _logger.error("max value is not unique for group {0}".format(chn_name))


# noinspection PyUnresolvedReferences
def output(source, target, csv):
    df = pd.read_parquet(source)
    df["date"] = to_unixtime(df.index)
    agc_mask = df["机组实际负荷"] >= 200
    df = df[agc_mask]
    df["_sum"] = 0
    A = 'A给煤机瞬时给煤量反馈'
    B = 'B给煤机瞬时给煤量反馈'
    C = 'C给煤机瞬时给煤量反馈'
    D = 'D给煤机瞬时给煤量反馈'
    E = 'E给煤机瞬时给煤量反馈'
    F = 'F给煤机瞬时给煤量反馈'
    for col in [A, B, C, D, E, F]:
        df[col] = df[col].fillna(0)
        df['_sum'] += df[col]
    # or_mask = df[col] >= 1
    sum_mask = df['_sum'] >= 1
    df = df[sum_mask]
    del df['_sum']
    df = df[(df[A] >= 1) | (df[B] >= 1) | (df[C] >= 1) | (df[D] >= 1) | (df[E] >= 1) | (df[F] >= 1)]
    points = []
    chn = read_csv(csv, sep=",", usecols=[2, 3])
    for _, row in chn.iterrows():
        _, point = row
        points.append(point)
    df = df[points]
    df.round(4).ffill().bfill().to_parquet(target)
    df.round(4).ffill().bfill().to_csv(target + ".csv", index=False)


# noinspection PyUnresolvedReferences
def calculate_extra(source, target):
    df = pd.read_parquet(source)
    from calculate import add_power, add_extra_coal
    add_power(df)
    add_extra_coal(df)
    df.to_parquet(target)


def concat_sensors(source, target, csv):
    df = read_csv(csv, sep=",", usecols=[3, 8])
    result = None
    """
    assert语句的格式是【assert 表达式，返回数据】，当表达式为False时则触发AssertionError异常.
    concat的文件是否存在
    """
    assert all_exists([join_path(source, str(rn)) for rn in range(len(df))])
    for rn, row in df.iterrows():
        chn_name, _ = row
        sensor = read_parquet(join_path(source, str(rn)), metadata=_date_value_frame)
        if result is None:
            result = sensor
            result.columns = [chn_name]
        else:
            # noinspection PyUnresolvedReferences
            result[chn_name] = sensor
    result.to_parquet(target)


def process_coal(target, csv, coal_hdf):
    coal_df = pd.read_hdf(coal_hdf, "table")
    df = read_csv(csv, sep="\\s+")
    for rn, row in df.iterrows():
        kks_names, chn_name, flag = row
        if kks_names.startswith("CoalQuality."):
            # noinspection PyUnresolvedReferences
            result = coal_df[chn_name]
            result.index.name = "date"
            result.name = "value"
            pd.DataFrame(result).to_parquet(os.path.join(target, str(rn)))


def process_feature(source, source2, target, csv):
    def select_source(p):
        if os.path.exists(os.path.join(source2, p)):
            return os.path.join(source2, p)
        elif os.path.exists(os.path.join(source, p)):
            return os.path.join(source, p)
        else:
            raise RuntimeError("{0} not found in {1} or {2}".format(p, source2, source))

    def re_re_sample(p):
        return p.resample(rule="20S").mean()

    df = read_csv(csv, sep=",", usecols=[3, 8])
    for rn, row in df.iterrows():
        chn_name, kks_names = row
        _logger.info("processing feature {0}".format(chn_name))
        # if kks_names.startswith("CoalQuality."):
        #   continue
        tgt = os.path.join(target, str(rn))
        if pd.isnull(kks_names):
            _logger.info("len is 0 --> kks_names {0}".format(kks_names))
        else:
            points = kks_names.split(" ")
            if len(points) == 1:
                src = select_source(kks_names)
                result = read_parquet(src, metadata=_date_value_frame)
                result = re_re_sample(result).ffill().bfill()
            else:
                if len(points) == 2:
                    agg = mean()
                elif len(points) == 3:
                    agg = median()
                elif len(points) >= 4:
                    agg = mean(except_min=True, except_max=True)
                else:
                    raise TypeError("kks_names {0} split have problem".format(kks_names))
                # fail fast
                files = [select_source(src) for src in points]
                for src in zip(points, files):
                    point, file = src
                    agg.offer(
                        re_re_sample(read_parquet(file, metadata=_date_value_frame)).ffill().bfill())
                result = agg.result()
        to_parquet(result, tgt)


def batch_waning(source, target, csv):
    df = read_csv(csv, sep=" ", usecols=[0, 1, 2, 3])
    for rn, row in df.iterrows():
        points, floor, ceil, flag = row
        for point in points.split(","):
            # noinspection PyTypeChecker
            df: pDataFrame = read_parquet(join_path(source, point), metadata=_date_value_frame)
            if not np.isnan(floor):
                floor_mask = df.value >= floor
                df = df[floor_mask]
            if not np.isnan(ceil):
                ceil_mask = df.value <= ceil
                df = df[ceil_mask]
            # todo re_sample后未经fill_na，所以这里无法区分fill/drop
            df.to_parquet(os.path.join(target, point))


# .toPandas().round(4).to_csv(target)
'''
        .where('date < "2017-06-11" and date > "2017-07-11" or date < "2018-03-22" and date > "2018-04-14" '
               'or date < "2018-05-08" and date > "2018-05-24" or date < "2018-07-25" and date > "2018-09-25" '
               'or date < "2019-04-30" and date > "2019-05-19" '
               'or date != "2017-09-23" or date != "2017-10-05" or date != "2019-01-20" ') \
'''
def describe(source, target, schema, key):
    return default_session().read.format(schema).parquet(source)\
        .withColumn("lag_value", fn.lag("value").over(Window.partitionBy(key).orderBy("date")))\
        .groupBy(key) \
        .agg(
        fn.max("value").alias("max"),
        fn.min("value").alias("min"),
        fn.avg("value").alias("avg")
    ).toPandas().round(4).to_csv(target)
    #).write.parquet("/export/grid/12/data/gxqing/njdata/describe/parquet")
"""
        fn.count("*").alias("count"),
        fn.stddev_samp("value").alias("stddev_samp"),
        fn.count(fn.when(fn.col("lag_value") == fn.col("value"), 1).otherwise(None)).alias("lag_count"),
        fn.expr("percentile_approx(value, 0.25)").alias("low_quartile"),
        fn.expr("percentile_approx(value, 0.75)").alias("high_quartile"),
"""


def re_sample_all(source, target, boundary, schema, up_time):
    with os.scandir(source) as files:
        points = [f.name for f in files]

    do_patch_spark()

    def re_sample_one(file):
        floor, ceil = boundary.get(file)
        _logger.info(("re_sampling {0}".format(file)))
        df = (data_frame(os.path.join(source, file), schema=schema, source_format="parquet", engine="spark")
              .ceilColumn("value", ceil, 1)
              .floorColumn("value", floor, 1)
              .select("date", "value")
              .where(fn.col("date").between(fn.to_timestamp(fn.lit(up_time.get("start"))),
                                            fn.to_timestamp(fn.lit(up_time.get("end")))))
              )
        _re_sample_s(df,
                     seconds=20,
                     write=True,
                     writer="spark",
                     fmt="parquet",
                     fname=join_path(target, file)
                     )

    from concurrent.futures import ThreadPoolExecutor as Pool
    from concurrent.futures import as_completed
    from heos.tools.misc import try_catch
    with Pool(max_workers=10) as executor:
        futures = [executor.submit(try_catch, re_sample_one, file=file) for file in points]
        error = False
        for future in as_completed(futures):
            succeed, args, kwargs, r = future.result()
            if not succeed:
                error = True
                from pyspark.sql.utils import CapturedException
                if isinstance(r, CapturedException):
                    _logger.error(
                        "error when re_sample, args {args}/kwargs {kwargs}{sep}desc {desc}{sep}stackTrace{stackTrace}"
                            .format(args=args, kwargs=kwargs, sep=os.sep, desc=r.desc, stackTrace=r.stackTrace))
                else:
                    _logger.error(
                        "error when re_sample,args {0}/kwargs {1}/exception {2}".format(str(args), str(kwargs), str(r)))
        if error:
            raise RuntimeError("error when re_sample")


def process_coal_quality(csv, source, target):
    df = read_csv(csv, usecols=[3, 8], sep=",")
    data = load_sensor(source, sep=",", index_col="date")
    for rn, row in df.iterrows():
        chn_name, kks_name = row
        tgt = join_path(target, kks_name)
        data[chn_name].name = "value"
        pdf_quality = pd.DataFrame(data[chn_name])
        pdf_quality["status"] = 0
        pdf_quality["point"] = kks_name
        # todo 输出的是什么数据
        '''
        selectExpr(*expr)
        Projects a set of SQL expressions and returns a new DataFrame.
        This is a variant of select() that accepts SQL expressions.
        '''
        to_parquet(pdf_quality, tgt, )
        # sdf_quality = koalas.DataFrame(pdf_quality).to_spark().selectExpr(
        #     "date", "{} as value".format(kks_name), "status", "point")
        # sdf_quality.write.parquet(tgt)


def convert_csv_to_parquet(source, target, schema):
    with os.scandir(source) as tree:
        files = list(tree)
    file_names = [os.path.join(target, f.name) for f in files]
    # assert not any_exists(file_names)
    for file in files:
        _logger.info("converting: {0}".format(file.name))
        point = file.name.replace(".csv", "")
        csv_to_parquet(
            source=file.path,
            target=os.path.join(target, point),
            schema=schema,
            partitions=5,
            lit_cols={"point": point},
            #options={"timestampFormat": date_format_java}
            date_format="%Y-%m-%d %H:%M:%S"
        )


"""
关于merge_splits（）参数说明：
测试参数：limit=100000
参数：group_selector=lambda entry: entry.name.split("_")[0].replace("/", "_").replace("/", "").replace("-", "_").replace("-", ""),
在20191211进行修改。新到的古交数据存在点名里面有下换线。单纯用.split("_")[0]，无法提取出点名。
"""


def merge_splits(source, target):
    merge(
        source=source,
        target=target,
        group_selector=lambda entry: entry.name.split("_20")[0].replace("/", "_").replace("/", "").replace("-", "_").replace("-", ""),
        file_filter=lambda entry: entry.name.find("_") > -1,
        encoding_rep=""
    )


def check_point_config():
    df = read_csv("data_config_lf.csv", sep=",", usecols=[3, 8])
    data = []
    for _, row in df.iterrows():
        _, kks_name = row
        if kks_name.find(' ') >= 0:
            for point in kks_name.split(" "):
                data += [[point]]
        else:
            data += [[kks_name]]
    df_point= pd.DataFrame(data, columns=["name"])
    df_point.to_csv("/export/grid/12/data/gxqing/njdata/min_max_lf.csv", index=False)


# master="local[128]"
_logger = logging.getLogger(__name__)
_init_spark = partial(init_spark_local, local_dir="/export/grid/12/data/gxqing/gujdata/temp", driver_memory="32g")
_date_value_frame = pd.DataFrame(columns=["value"],
                                 index=pd.DatetimeIndex([], dtype='datetime64[ns]', name='date', freq=None))

if __name__ == "__main__":
    main(
        do_describe=True
    )
