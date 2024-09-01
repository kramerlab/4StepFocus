from pathlib import Path


class Logger:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def log(self, text: str, print_to_console: bool = True):
        if print_to_console:
            print(text)

        # Open the log_file file in append mode ('a')
        with (open(self.output_path / "log.txt", 'a', encoding='utf-8', errors='replace') as log_file):
            log_file.write(text + "\n")

    def save_result(self, question_id: int, question: str, final_answer: str, ground_truth: str):
        with (open(self.output_path / "results.csv", 'a', encoding='utf-8', errors='replace') as result_file):
            result_file.write(f"{question_id};{question};{final_answer};{ground_truth}\n")
