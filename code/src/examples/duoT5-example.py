from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load DuoT5 model and tokenizer
model_name = "castorini/duot5-base-msmarco"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the query and documents
query = "What is the capital of France?"
document0 = "Paris is the capital and most populous city of France."
document1 = "Lyon is a major city in France known for its cuisine."

# Prepare the input in the DuoT5 format
input_text = f"Query: {query} Document0: {document0} Document1: {document1}"

# Tokenize the input
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate output
outputs = model.generate(inputs)

# Decode the output
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Interpret the result
if result == "Document0":
    print("Document 0 is more relevant.")
elif result == "Document1":
    print("Document 1 is more relevant.")
else:
    print(f"Unexpected result: {result}")
