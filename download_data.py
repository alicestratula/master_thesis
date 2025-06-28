import json
import os

import openml
import numpy as np



suites = [379]

def main():
    for suite_id in suites:
        benchmark_suite = openml.study.get_suite(suite_id)
        tasks = benchmark_suite.tasks

        os.makedirs("task_indices", exist_ok=True)
        np.save(f"task_indices/{suite_id}_task_indices.npy", tasks)

        names = []

        for task_id in benchmark_suite.tasks:
            task = openml.tasks.get_task(task_id)  # download the OpenML task
            dataset = task.get_dataset()
            name = dataset.name
            names.append(name)

            X, y, categorical_indicator, _ = dataset.get_data(
                target=task.target_name
            )

            path = f"original_data/{suite_id}_{task_id}"
            os.makedirs(path, exist_ok=True)

            X, y, categorical_indicator, _ = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            X.to_csv(os.path.join(path, f"{suite_id}_{task_id}_X.csv"), index=False)
            y.to_csv(os.path.join(path, f"{suite_id}_{task_id}_y.csv"), index=False)

            np.save(os.path.join(path, f"{suite_id}_{task_id}_categorical_indicator.npy"), categorical_indicator) 

        with open(f"task_indices/{suite_id}_task_names.json", "w") as f:
            json.dump(names, f)


if __name__ == '__main__':
    main()