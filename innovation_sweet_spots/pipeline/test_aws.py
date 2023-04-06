"""
Metaflow to test sending a job to AWS batch

Usage (running from the terminal):
python innovation_sweet_spots/pipeline/test_aws.py run
"""
from metaflow import FlowSpec, step, batch
from innovation_sweet_spots import PROJECT_DIR
import pandas as pd

# Outputs parameters
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_NAME = "test_aws_fibonacci"


class Aws_test(FlowSpec):
    @step
    def start(self):
        """Initialise the flow"""
        self.next(self.run_test)

    @batch(
        queue="job-queue-GPU-nesta-metaflow",
        image="metaflow-pytorch",
        # Queue gives p3.2xlarge, with:
        gpu=1,
        memory=60000,
        cpu=8,
    )
    @step
    def run_test(self):
        """Calculate the first 10000 fibonacci numbers"""
        self.fibs = [0, 1]
        for i in range(10000):
            self.fibs.append(self.fibs[-1] + self.fibs[-2])
        self.next(self.save_outputs)

    @step
    def save_outputs(self):
        """Saves outputs"""
        # Save list to a csv file
        pd.DataFrame(self.fibs).to_csv(OUTPUT_DIR / f"{OUTPUT_NAME}.csv", index=False)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    Aws_test()
