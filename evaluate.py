import json, os

result_file_name = "results.json"

# load each students results
with open(os.path.join(result_file_name)) as f:
    student_results_dict = json.load(f)

# load test info
with open(os.path.join('ground_truth_1000.json')) as f:
    test_info_dict = json.load(f)

def compute_accuracy(ground_truth, results, name="student_name", prec=1):
    co, total = 0.0, 0.0
    for res in results:
        p_idx = results[res]
        g_idx = ground_truth[res]
        if g_idx==p_idx:
            co += 1.0
        total += 1.0

    print ("Accuracy of {} results on {} images is: {}%.".format(name, int(total), round(100*co/total, prec)))

compute_accuracy(test_info_dict, student_results_dict, name=result_file_name.split("_")[0])