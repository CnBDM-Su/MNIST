from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#initialization
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
#read
training = spark.read.format("csv").option("header","true").option("inferSchema","true").load("s3://aws-logs-799591416097-us-east-1/mnist_train.csv")
testing = spark.read.format("csv").option("header","true").option("inferSchema","true").load("s3://aws-logs-799591416097-us-east-1/mnist_test.csv")
#convert csv to libsvm
assembler = VectorAssembler(inputCols=["pixel{0}".format(i) for i in range(784)], outputCol="features")
labelPoints = assembler.transform(training).select("label","features")
labelPoints2 = assembler.transform(testing).select("label","features")

#build model
#Random Forest Classifier
lr = RandomForestClassifier(labelCol="label",featuresCol="features")

#NaiveBayes
#lr = NaiveBayes(smoothing=1.0,modelType="multinomial")

pipeline = Pipeline(stages=[lr])

#Parameter Grid
paramGrid = ParamGridBuilder() \
    .addGrid(lr.numTrees, range(3, 10)) \
    .addGrid(lr.maxDepth, range(4, 10)) \
    .build()

#Cross Validator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)

# training model
lrModel = crossval.fit(labelPoints)
#lrModel = pipeline.fit(labelPoints)
#training accuracy
predictions = lrModel.transform(labelPoints)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol = "label", metricName="accuracy")
print("Accuracy:" + str(evaluator.evaluate(predictions)))


#testing accuracy
predictions2 = lrModel.transform(labelPoints2)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol = "label", metricName="accuracy")
print("Accuracy:" + str(evaluator.evaluate(predictions2)))

predictions.rdd.saveAsTextFile("s3://aws-logs-799591416097-us-east-1/result/trainModel")
predictions2.rdd.saveAsTextFile("s3://aws-logs-799591416097-us-east-1/result/testModel")
sc.parallelize(["Training Accuracy:"+str(evaluator.evaluate(predictions))+", Testing Accuracy:"+str(evaluator.evaluate(predictions2))]).saveAsTextFile("s3://aws-logs-799591416097-us-east-1/result/accuracy")