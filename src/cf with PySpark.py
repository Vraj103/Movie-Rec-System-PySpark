# 1.3 Item based collaborative filtering works the best as of now for Pandas.

data = pd.read_csv("define/your/path")

# Step 2: Now, without downsampling the data, we use collaborative filtering with Spark.

spark = SparkSession.builder \
    .appName("UserUserCF") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.memory.fraction", "0.8") \
    .getOrCreate()

# Assuming 'df' is your dataset in a Pandas DataFrame format
# Convert it to a Spark DataFrame
spark_df = spark.createDataFrame(data[['userId', 'id', 'rating']])

# Show schema to confirm successful load
spark_df.printSchema()

# %% [code] {"id":"G6NRzl3JoMIL"}
ratings_df = spark_df.select("userId", "id", "rating") \
                     .withColumn("userId", spark_df["userId"].cast("integer")) \
                     .withColumn("movieId", spark_df["id"].cast("integer")) \
                     .withColumn("rating", spark_df["rating"].cast("float"))

# %% [code] {"id":"lqf3vTfkuKnF"}
# Split the data
train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

# %% [code] {"id":"WcDaPKgcuVhQ","outputId":"ee4ac344-a139-4463-a348-81ae80f1c05a"}
total_time_spark = 0
# Configure ALS for user-user CF
als = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

start_time = time.time()
# Train ALS model
als_model = als.fit(train_df)
end_time = time.time()
total_time_spark += (end_time - start_time)
print(f"Training time: {total_time_spark:.2f} seconds")

# %% [code] {"id":"fWO-TVj7vrn7","outputId":"404f8a2f-91e7-485a-fb58-cfb7bc646206"}
from pyspark.ml.evaluation import RegressionEvaluator

# Make predictions
predictions = als_model.transform(test_df)

# Evaluate RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error (RMSE): {rmse}")

# %% [code] {"id":"LpPVghg_9xTq","outputId":"d0a0ef27-86e1-4e13-9553-93a11378b2ed"}
user_id = 900
item_id = 46
pred = model.predict(user_id, item_id)
print(f"Prediction for user {user_id} on item {item_id}: {pred.est}")
