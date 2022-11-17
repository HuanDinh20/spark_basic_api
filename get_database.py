import pathlib
import pickle

from pyspark.context import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, min, max, first, last, var_pop, stddev_pop, var_samp, stddev_samp
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *



class AggregateFeature:
    def __init__(self, filepath, appName="Default Name", view_name="temp_view"):
        self.conf, self.sc, self.spark = self.create_spark_basic(appName)
        self.file_format = pathlib.Path(filepath).suffix
        self.data_frame = self.read_csv_spark(filepath, self.file_format)
        self.data_frame.createOrReplaceTempView(view_name)

    @staticmethod
    def create_spark_basic(appName: str):
        """
        :param appName: Name of your app, string
        :return:
        """
        conf = SparkConf().setAppName(appName)
        sc = SparkContext()
        spark = SparkSession.builder.appName(
            appName).config(conf=conf).getOrCreate()
        return conf, sc, spark

    def read_csv_spark(self, filepath, file_format="csv"):
        if file_format == ".csv":
            df = self.spark.read.csv(filepath, header=True, inferSchema=True)
        elif file_format == ".json":
            df = self.spark.read.json(filepath)
        elif file_format == ".parquet":
            df = self.spark.read.parquet(filepath)
        else:
            print(f"Not supported {file_format}, currently supported: csv, json, parquet")
            return None
        return df

    def count_function(self, column: str):
        count_data = self.data_frame.select(count(column))
        return count_data

    def average_function(self, column):
        avg_data = self.data_frame.select(avg(column))
        return avg_data

    def get_first_last(self, column):
        first_last_df = self.data_frame.select(first(column), last(column))
        return first_last_df

    def get_avg(self, column):
        min_df = self.data_frame.select(min(column))
        max_df = self.data_frame.select(max(column))
        return min_df, max_df

    def get_var(self, column):
        try:
            var_df = self.data_frame.select(var_pop(column), var_samp(column))
            return var_df
        except Exception as e:
            print(e)
            return None

    def get_std(self, column):
        try:
            var_df = self.data_frame.select(stddev_samp(column), stddev_pop(column))
            return var_df
        except Exception as e:
            print(e)
            return None


class DMLFeatures(AggregateFeature):
    def __init__(self, filepath, appName="Deault Name", view_name="temp_view"):
        super().__init__(filepath, appName, view_name)
        self.view_name = view_name

    def group_by_and_show(self, column):
        self.data_frame.groupby(column).count().show()

    def roll_up_and_show(self, column):
        self.data_frame.rollup(column).count().show()

    def cube_show(self, column):
        self.data_frame.cube(column).count().show()

    def query(self, your_queries):
        self.spark.sql(f"{your_queries} FROM {self.view_name}").show()

    def show_table(self):
        self.spark.sql("SHOW TABLES").show()

class MLBuildFeature:
    def __init__(self, dataset, target, algorithm= DecisionTreeClassifier):
        self.dataset = dataset
        self.target = target
        self.algorithm = algorithm
        self.encoded_label_df, self._target_label_name = self.label_encode()
        # self._df_encoded, self._target_label_onehot_name = self.one_hot_encoder(self._encoded_label)
        self._vector_assembler = self.create_training_features()
        self.train_set, self.val_set = self.create_training_test_dataset()
        self.clf = self.create_model()
        self.predictions, self.accuracy = self.evaluate_model()

    def label_encode(self, target_encode_name=None):
        if target_encode_name is None:
            target_encode_name = self.target + "_encoded"
        indexer = StringIndexer(inputCol=self.target, outputCol=target_encode_name)
        encoded_label_df = indexer.fit(self.dataset).transform(self.dataset)
        return encoded_label_df, target_encode_name

    def one_hot_encoder(self, encoded_label_df, target_encode_name=None):
        if target_encode_name is None:
            target_encode_name = self.target + "_onehot_encoded"
        onehot_encoder = OneHotEncoder(inputCol=self.target, outputCol=target_encode_name)
        df_encoded = onehot_encoder.fit(encoded_label_df).transform(encoded_label_df)
        return df_encoded, target_encode_name

    def create_training_features(self, training_feature: list = None, outputCol="features"):
        if training_feature is None:
            training_feature = []
            for col in self.dataset.columns:
                if col not in [self._target_label_name,  self.target]:
                    training_feature.append(col)
        vector_assembler = VectorAssembler(inputCols=training_feature, outputCol=outputCol)
        return vector_assembler

    def create_training_test_dataset(self, training_size=0.7, targe_encoding="label_encode"):
        df = self._vector_assembler.transform(self.encoded_label_df)
        features = self._vector_assembler.getOutputCol()
        print("**"*10)
        print(self.encoded_label_df)
        model_df = df.select(features, self._target_label_name)
        train_set, val_set = model_df.randomSplit([training_size, 1 - training_size])
        return train_set, val_set

    def create_model(self, _target_label_name=None):
        if _target_label_name is None:
            _target_label_name = self._target_label_name
        clf = self.algorithm(labelCol=_target_label_name).fit(self.train_set)
        return clf

    def evaluate_model(self, test_set=None, _target_label_name=None):
        if test_set is None:
            test_set = self.val_set
        pred = self.clf.transform(test_set)
        if _target_label_name is None:
            _target_label_name = self._target_label_name
        evaluator = MulticlassClassificationEvaluator(labelCol=_target_label_name, metricName="accuracy")
        acc = evaluator.evaluate(pred)
        return pred, acc

    @staticmethod
    def save_model(model, filepath="./model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(model, f)






