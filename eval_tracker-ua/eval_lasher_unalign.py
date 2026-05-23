from metrics import LasHeR_Unalign

lasher = LasHeR_Unalign(
    gt_path=f'eval_tracker-ua/metrics/gt_file/LasHeR_Unalign/annos/',
    seq_name_path=f"eval_tracker-ua/metrics/gt_file/LasHeR_Unalign/lashertest.txt")

"""
LasHeR have 3 benchmarks: PR, NPR, SR
"""
name = 'output/tracking_results/baseline/lasher_baseline_ep20_'
# Register your tracker
lasher(
    tracker_name="tracker2",
    result_path=f"{name}/LasHeR-Unaligned", 
    bbox_type="ltwh")

pr_dict = lasher.PR()
npr_dict = lasher.NPR()
sr_dict = lasher.SR()


print(pr_dict["tracker2"][0])
print(npr_dict["tracker2"][0])
print(sr_dict["tracker2"][0])

# lasher.draw_plot(metric_fun=lasher.PR)
# lasher.draw_plot(metric_fun=lasher.NPR)
# lasher.draw_plot(metric_fun=lasher.SR)