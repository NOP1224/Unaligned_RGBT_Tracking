from metrics import LUART

luart = LUART(gt_path=f'eval_tracker-ua/metrics/gt_file/LUART/luart_gt/rgb',
            seq_name_path=f"eval_tracker-ua/metrics/gt_file/LUART/testing_list.txt")

"""
LasHeR have 3 benchmarks: PR, NPR, SR
"""
# Register your tracker
luart(
    tracker_name="tracker1",
    result_path="output/tracking_results/baseline/lasher_baseline_ep20_/LUART", 
    bbox_type="ltwh")

pr_dict = luart.PR()
npr_dict = luart.NPR()
sr_dict = luart.SR()

print(pr_dict["tracker1"][0])
print(npr_dict["tracker1"][0])
print(sr_dict["tracker1"][0])

# print(pr_dict["tracker2"][0])
# print(npr_dict["tracker2"][0])
# print(sr_dict["tracker2"][0])

# lasher.draw_plot(metric_fun=lasher.PR)
# lasher.draw_plot(metric_fun=lasher.NPR)
# lasher.draw_plot(metric_fun=lasher.SR)