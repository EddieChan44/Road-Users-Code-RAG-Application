# test.py
import os
import sys
import time
import logging
import pandas as pd
from chat import RoadUsersCodeChat

# Suppress detailed logging output
logging.getLogger("langchain_community.embeddings.ollama").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)

class CustomProgress:
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.n = 0
        self.desc = desc
        print(f"{desc}: 0/{total} [0%]", end="")
        
    def update(self):
        self.n += 1
        progress = int(self.n / self.total * 100)
        print(f"\r{self.desc}: {self.n}/{self.total} [{progress}%]", end="")
        sys.stdout.flush()
        
    def close(self):
        print()

def run_tests():
    print("Initializing RAG model...")
    try:
        chat = RoadUsersCodeChat()
    except Exception as e:
        print(f"Error initializing RAG model: {e}")
        return

    # Check if the test_case CSV file exists
    if not os.path.exists("test_case.csv"):
        print("Test case file 'test_case.csv' not found.")
        return
    
    print("Loading test cases...")
    test_cases = pd.read_csv("test_case.csv", header=None)
    test_cases.columns = ["question", "expected_answer"]
    
    results = []
    total_time = 0.0
    
    print(f"Testing {len(test_cases)} questions...")
    progress = CustomProgress(len(test_cases), "Testing questions")
    
    for idx, row in test_cases.iterrows():
        question = row["question"]
        expected = row["expected_answer"]
        
        start_time = time.time()
        try:
            actual_answer = chat.process_query(question)
        except Exception as ex:
            actual_answer = f"Error processing query: {ex}"
        end_time = time.time()
        
        response_time = end_time - start_time
        total_time += response_time
        
        results.append({
            "question_id": idx + 1,
            "question": question,
            "expected_answer": expected,
            "actual_answer": actual_answer,
            "response_time": response_time
        })
        progress.update()
    
    progress.close()
    
    # Calculate average response time
    avg_response_time = total_time / len(test_cases) if len(test_cases) > 0 else 0
    print(f"\nAverage response time: {avg_response_time:.2f} seconds")
    
    # Save detailed results to CSV for review
    results_df = pd.DataFrame(results)
    results_csv = "rag_test_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Detailed results saved to '{results_csv}'")
    
    # Display comparison results for manual review
    print("\n" + "=" * 80)
    print(" " * 30 + "MANUAL COMPARISON")
    print("=" * 80 + "\n")
    
    print("OVERVIEW OF ALL TEST QUESTIONS:")
    for idx, row in results_df.iterrows():
        question_num = row['question_id']
        question = row['question']
        print(f"{question_num}. {question}")
    
    print("\n" + "=" * 80)
    print(" " * 25 + "DETAILED COMPARISON RESULTS")
    print("=" * 80)
    
    for idx, row in results_df.iterrows():
        print(f"\n🔹 QUESTION {row['question_id']}: {row['question']}")
        print(f"\n📝 EXPECTED ANSWER:")
        print(f"   {row['expected_answer']}")
        print(f"\n🤖 ACTUAL ANSWER:")
        print(f"   {row['actual_answer']}")
        print(f"\n⏱️  RESPONSE TIME: {row['response_time']:.2f} seconds")
        print("\n" + "-" * 80)
    
    print("\n" + "=" * 80)
    print(f"TESTING COMPLETE: {len(test_cases)} questions evaluated")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    print("=" * 80)
    
    # Clear the chat history after testing
    chat.clear_history()
    print("\nChat history cleared.")

if __name__ == '__main__':
    run_tests()