#let's do our recommendations based on user-user, item-item and Matrix Factorization techniques using Model implementations but here, we'll be using downsampling (using 20% of the data) then batch processing to make sure the ram doesn't crash.
data = pd.read_csv("define/your/path"))
# 1st Alternative
# Downsample the data to 10% of the original size
downsampled_df = data.sample(frac=0.5, random_state=1001)

# Check the size of the downsampled dataset
print(f"Downsampled dataset size: {downsampled_df.shape}",downsampled_df.info())

# Define batch size, e.g., 10,000 rows at a time (I've done it 1.5k at the time as the ram didn't crash.)
batch_size = 1500

# Reader for Surprise with the downsampled data's rating range
reader = Reader(rating_scale=(downsampled_df['rating'].min(), downsampled_df['rating'].max()))

# Initialize user-user collaborative filtering model
#sim_options = {'name': 'cosine', 'user_based': False} #user-based is true if the user wants user-user interactions and false if item item
model = SlopeOne()
#---User-User
#KNNBasic = 0.6764
#KNNWithMeans = 0.0656----192.69 seconds
#KNNWithZScore = 0.0735
#---Item-Item
#KNNWithZScore = 0.0667
#KNNWithMeans = 0.0618----62.32 seconds
#KNNBasic = 0.1038
#---Model Based
#SVD = 0.7047
#CoClustering = 0.8093
#NMF = 0.1739----Also, 148.05 Seconds
#SlopeOne = 0.0210 best so far #71.48 seconds
#BaselineOnly = 0.9077
#SVDpp = 0.7243

# List to collect predictions across all batches
all_predictions = []

# %% [code] {"id":"DdQDuQqLoQcM","outputId":"9a4042e9-6adb-4691-ab41-749e9e5bfa19","jupyter":{"outputs_hidden":true}}
# Calculate the number of batches needed
num_batches = len(downsampled_df) // batch_size + (1 if len(downsampled_df) % batch_size != 0 else 0)
total_time = 0

for i in range(num_batches):
    # Define start and end index for the batch
    start = i * batch_size
    end = start + batch_size

    # Get a batch from the downsampled DataFrame
    batch_df = downsampled_df.iloc[start:end][['userId', 'id', 'rating']]

    # Convert batch to Surprise dataset format
    data_batch = Dataset.load_from_df(batch_df, reader)
    trainset = data_batch.build_full_trainset()

    # Record start time
    start_time = time.time()

    # Train the model on the current batch
    model.fit(trainset)

    # Record end time
    end_time = time.time()

    runtime = end_time - start_time

    total_time += runtime

    # Create test set for predictions (using the same batch as test set for simplicity)
    testset = trainset.build_testset()
    predictions = model.test(testset)

    # Collect predictions
    all_predictions.extend(predictions)

print(f"Total training time across all batches: {total_time:.2f} seconds")

#Root mean squared error
rmse = accuracy.rmse(all_predictions)
print(f"RMSE across all batches (downsampled): {rmse}")


scattter_data = pd.read_csv("define/your/path")
scattter_data

from matplotlib import pyplot as plt
import seaborn as sns
figsize = (12, 1.2 * len(scattter_data['Type'].unique()))
plt.figure(figsize=figsize)
sns.violinplot(scattter_data, x='Value', y='Type', inner='stick', palette='Dark2')
sns.despine(top=True, right=True, bottom=True, left=True)

