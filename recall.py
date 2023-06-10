def calculate_evaluation_metrics(evaluation_file_path):
    # Read the evaluation file
    evaluation_data = []

    with open(evaluation_file_path, 'r') as evaluation_file:
        for line in evaluation_file:
            line = line.strip()
            if line:
                evaluation_data.append(line)

    # Process the evaluation data
    query_id = None
    num_relevant = 0
    num_returned = 0
    total = 0
    average_precision_sum = 0
    query_count = 0
    precision_list = []  # List to store precision values
    recall_list = []  # List to store recall values
    reciprocal_rank_sum = 0  # Sum of reciprocal ranks

    for line in evaluation_data:
        if line.startswith("Query ID:"):
            # Check if there are previous query results to process
            if query_id is not None:
                # Calculate precision and recall for the previous query
                precision = num_relevant / num_returned if num_returned > 0 else 0.0
                recall = num_relevant / total if total > 0 else 0.0

                # Add precision and recall to the lists
                precision_list.append(precision)
                recall_list.append(recall)

                # Calculate average precision for the previous query and add it to the sum
                average_precision = precision * num_relevant / total if total > 0 else 0.0
                average_precision_sum += average_precision

                # Calculate reciprocal rank for the previous query and add it to the sum
                reciprocal_rank = 1 / (num_returned + 1) if num_returned > 0 else 0.0
                reciprocal_rank_sum += reciprocal_rank

                # Reset the values for the next query
                num_relevant = 0
                num_returned = 0

            # Extract the query ID for the current query
            query_id = int(line.split(":")[1].strip())
            query_count += 1

        elif line.startswith("Relevant Documents:"):
            # Extract the number of relevant documents for the current query
            num_relevant = int(line.split(":")[1].strip())

        elif line.startswith("Total Returned Documents:"):
            # Extract the number of returned documents for the current query
            num_returned = int(line.split(":")[1].strip())

        elif line.startswith("Total:"):
            # Extract the total number of documents for the current query
            total = int(line.split(":")[1].strip())

    # Calculate precision and recall for the last query in the file
    precision = num_relevant / num_returned if num_returned > 0 else 0.0
    recall = num_relevant / total if total > 0 else 0.0

    # Add precision and recall to the lists
    precision_list.append(precision)
    recall_list.append(recall)

    # Calculate average precision for the last query and add it to the sum
    average_precision = precision * num_relevant / total if total > 0 else 0.0
    average_precision_sum += average_precision

    # Calculate reciprocal rank for the last query and add it to the sum
    reciprocal_rank = 1 / (num_returned + 1) if num_returned > 0 else 0.0
    reciprocal_rank_sum += reciprocal_rank

    # Calculate Mean Average Precision (MAP)
    map_score = average_precision_sum / query_count if query_count > 0 else 0.0

    # Calculate Mean Reciprocal Rank (MRR)
    mrr_score = reciprocal_rank_sum / query_count if query_count > 0 else 0.0

    # Return the evaluation metrics
    evaluation_metrics = {
        'MAP': map_score,
        'MRR': mrr_score,
        'Precision': precision_list,
        'Recall': recall_list
    }
    return evaluation_metrics

# Usage example
evaluation_file_path = r"C:\Users\User\Desktop\relevanttt_num_queries.txt"
evaluation_metrics = calculate_evaluation_metrics(evaluation_file_path)

# Print precision, recall, and MRR for each query
for i, precision in enumerate(evaluation_metrics['Precision']):
    print(f"Query {i+1}: Precision = {precision}, Recall = {evaluation_metrics['Recall'][i]}, MRR = {evaluation_metrics['MRR']}")

# Print MAP and MRR
print("MAP:", evaluation_metrics['MAP'])
print("MRR:", evaluation_metrics['MRR'])
