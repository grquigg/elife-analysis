def extract_reviews_from_txt(in_dir, tasks, subset):
    reviews = {}
    with open(f"{in_dir}/{subset}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            review_id, sentence_index = example["identifier"].split("|||")
            if review_id not in reviews:
                reviews[review_id] = {
                    "sentences": [],
                    "labels": []
                }
            reviews[review_id]["sentences"].append(example["text"])
            reviews[review_id]["labels"].append([example[label] for label in ta>
    return reviews
